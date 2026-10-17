"""Simulated ops tools: deployments, pods, logs, rollbacks and scaling.

Nothing here touches a real cluster. Pods are derived from each deployment's current revision and
replica count: a healthy revision runs, a broken one crash-loops. Pod names hash the revision's
image, so they change after a rollback. Logs are canned per image in fixtures/logs.json, built from
Loghub samples by fixtures/make_fixtures.py.
"""

import hashlib
import json
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field

from gym_env.helpdesk import find_ticket
from gym_env.state import FIXTURES_DIR, Change, Deployment, Namespace, Revision, State
from gym_env.tools import Args, Tool, ToolError

DeploymentName = Annotated[str, Field(description="Deployment name, like payments.")]
ChangeTicket = Annotated[
    str | None,
    Field(
        pattern=r"^HD-\d+$",
        description="Required for changes in prod: the ID of an open helpdesk ticket for this change.",
    ),
]

_POD_ALPHABET = "bcdfghjklmnpqrstvwxz2456789"  # Kubernetes' alphabet for generated name suffixes


@lru_cache(maxsize=None)
def _logs() -> dict[str, list[str]]:
    return json.loads((FIXTURES_DIR / "logs.json").read_text())["logs"]


def _suffix(length: int, *parts: object) -> str:
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return "".join(_POD_ALPHABET[b % len(_POD_ALPHABET)] for b in digest[:length])


def _pods(namespace: str, deployment: Deployment) -> list[dict[str, Any]]:
    revision = deployment.current
    template = _suffix(4, namespace, deployment.name, revision.image)
    pods = []
    for i in range(deployment.replicas):
        pod = {
            "name": f"{deployment.name}-{template}-{_suffix(3, namespace, deployment.name, revision.image, i)}",
            "image": revision.image,
        }
        if revision.healthy:
            pod |= {"status": "Running", "ready": "1/1", "restarts": 0}
        else:
            restarts = 4 + int(hashlib.sha256(pod["name"].encode()).hexdigest(), 16) % 9
            pod |= {"status": "CrashLoopBackOff", "ready": "0/1", "restarts": restarts}
        pods.append(pod)
    return pods


def _find_deployment(state: State, namespace: str, name: str) -> Deployment:
    for deployment in state.namespaces[namespace]:
        if deployment.name == name:
            return deployment
    names = ", ".join(d.name for d in state.namespaces[namespace])
    raise ToolError(
        "not_found", f"Deployment {name!r} not found in {namespace}. Deployments in {namespace}: {names}."
    )


def _check_change_ticket(state: State, namespace: str, change_ticket: str | None) -> str | None:
    """Cross-API rule: a change in prod needs the ID of an open (unresolved) helpdesk ticket."""
    if namespace != "prod":
        return None
    if change_ticket is None:
        raise ToolError(
            "change_ticket_required",
            "Changes in the prod namespace require change_ticket: the ID of an open helpdesk ticket "
            "(status open or in_progress) that covers this change. Open one in the helpdesk if there "
            "isn't one, then retry with change_ticket set to its ID.",
        )
    try:
        ticket = find_ticket(state, change_ticket)
    except ToolError:
        raise ToolError(
            "invalid_change_ticket",
            f"change_ticket {change_ticket} does not exist. Prod changes need the ID of an existing open "
            "helpdesk ticket.",
        ) from None
    if ticket.status == "resolved":
        raise ToolError(
            "invalid_change_ticket",
            f"change_ticket {change_ticket} is resolved. Prod changes need an open ticket "
            "(status open or in_progress).",
        )
    return change_ticket


def _record(state: State, namespace: Namespace, deployment: Deployment, action: str,
            before: int, after: int, change_ticket: str | None) -> None:
    state.changes.append(
        Change(namespace=namespace, deployment=deployment.name, action=action, before=before,
               after=after, change_ticket=change_ticket, at=state.now)
    )


class ListDeploymentsArgs(Args):
    namespace: Namespace


def list_deployments(state: State, args: ListDeploymentsArgs) -> dict[str, Any]:
    """List the deployments in a namespace with their image, revision and ready replicas."""
    deployments = []
    for d in state.namespaces[args.namespace]:
        ready = d.replicas if d.current.healthy else 0
        deployments.append(
            {"name": d.name, "image": d.current.image, "revision": d.current_revision,
             "replicas": d.replicas, "ready": f"{ready}/{d.replicas}"}
        )
    return {"namespace": args.namespace, "deployments": deployments}


class GetPodsArgs(Args):
    namespace: Namespace
    deployment: DeploymentName


def get_pods(state: State, args: GetPodsArgs) -> dict[str, Any]:
    """List a deployment's pods with their status and restart counts. Pod names are needed for get_logs."""
    deployment = _find_deployment(state, args.namespace, args.deployment)
    return {"namespace": args.namespace, "deployment": deployment.name,
            "pods": _pods(args.namespace, deployment)}


class GetLogsArgs(Args):
    namespace: Namespace
    pod: str = Field(description="Pod name from get_pods, like payments-7d9f-x2k.")
    tail: int = Field(20, ge=1, le=200, description="Number of most recent lines to return.")


def get_logs(state: State, args: GetLogsArgs) -> dict[str, Any]:
    """Get the most recent log lines of a pod."""
    for deployment in state.namespaces[args.namespace]:
        for pod in _pods(args.namespace, deployment):
            if pod["name"] == args.pod:
                lines = _logs()[pod["image"]]
                return {"namespace": args.namespace, "pod": args.pod, "lines": lines[-args.tail:]}
    raise ToolError(
        "not_found",
        f"Pod {args.pod!r} not found in {args.namespace}. Pod names change after a rollback or "
        "restart; call get_pods(namespace, deployment) for the current names.",
    )


class RolloutHistoryArgs(Args):
    namespace: Namespace
    deployment: DeploymentName


def _revision_view(revision: Revision) -> dict[str, Any]:
    return revision.model_dump(exclude={"healthy"})


def rollout_history(state: State, args: RolloutHistoryArgs) -> dict[str, Any]:
    """List a deployment's revisions, oldest first, and which one is current. Rollbacks need a revision number from here."""
    deployment = _find_deployment(state, args.namespace, args.deployment)
    return {"namespace": args.namespace, "deployment": deployment.name,
            "current_revision": deployment.current_revision,
            "revisions": [_revision_view(r) for r in deployment.revisions]}


class RollbackArgs(Args):
    namespace: Namespace
    deployment: DeploymentName
    to_revision: int = Field(ge=1, description="Revision number from rollout_history.")
    change_ticket: ChangeTicket = None


def rollback(state: State, args: RollbackArgs) -> dict[str, Any]:
    """Roll a deployment back (or forward) to a revision from its rollout history.
    In prod this needs change_ticket, the ID of an open helpdesk ticket."""
    deployment = _find_deployment(state, args.namespace, args.deployment)
    target = deployment.revision(args.to_revision)
    if target is None:
        numbers = ", ".join(str(r.revision) for r in deployment.revisions)
        raise ToolError(
            "not_found",
            f"{args.deployment} has no revision {args.to_revision}. Revisions: {numbers}. "
            "See rollout_history.",
        )
    before = deployment.current_revision
    if before == args.to_revision:
        return {"deployment": deployment.name, "revision": before, "changed": False,
                "message": f"{deployment.name} is already at revision {before}; nothing changed."}
    ticket = _check_change_ticket(state, args.namespace, args.change_ticket)
    deployment.current_revision = args.to_revision
    _record(state, args.namespace, deployment, "rollback", before, args.to_revision, ticket)
    return {"deployment": deployment.name, "namespace": args.namespace, "changed": True,
            "from_revision": before, "to_revision": args.to_revision, "image": target.image,
            "change_ticket": ticket}


class ScaleArgs(Args):
    namespace: Namespace
    deployment: DeploymentName
    replicas: int = Field(ge=0, le=20, description="New number of replicas.")
    change_ticket: ChangeTicket = None


def scale(state: State, args: ScaleArgs) -> dict[str, Any]:
    """Set a deployment's number of replicas. In prod this needs change_ticket, the ID of an open helpdesk ticket."""
    deployment = _find_deployment(state, args.namespace, args.deployment)
    before = deployment.replicas
    if before == args.replicas:
        return {"deployment": deployment.name, "replicas": before, "changed": False,
                "message": f"{deployment.name} already has {before} replicas; nothing changed."}
    ticket = _check_change_ticket(state, args.namespace, args.change_ticket)
    deployment.replicas = args.replicas
    _record(state, args.namespace, deployment, "scale", before, args.replicas, ticket)
    return {"deployment": deployment.name, "namespace": args.namespace, "changed": True,
            "from_replicas": before, "to_replicas": args.replicas, "change_ticket": ticket}


TOOLS = [
    Tool("list_deployments", "ops", ListDeploymentsArgs, list_deployments),
    Tool("get_pods", "ops", GetPodsArgs, get_pods),
    Tool("get_logs", "ops", GetLogsArgs, get_logs),
    Tool("rollout_history", "ops", RolloutHistoryArgs, rollout_history),
    Tool("rollback", "ops", RollbackArgs, rollback),
    Tool("scale", "ops", ScaleArgs, scale),
]
