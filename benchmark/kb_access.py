"""Shared READ-ONLY Neo4j access for the A-series KG-intrinsic evaluations.

Connects with the raw `neo4j` driver using `Neo4jConfig` credentials — deliberately NOT via
`Neo4jClient`, because that loads the Qwen embedding model on connect (slow, unnecessary for
pure-Cypher diagnostics). Stored `embedding` properties are read straight off the nodes when an
A-series test needs vectors (e.g. A1 duplication), so no model load is ever required.

ALL queries here are read-only. The A-series harnesses never write to the shared KB (another
session may be building it); experiments that would mutate the graph (SEA re-evolve, ingestion)
live behind explicit opt-in guards in their own modules and are not invoked by the diagnostics.
"""

from __future__ import annotations

import contextlib

from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig


def _creds(cfg: Neo4jConfig) -> tuple[str, str, str, str]:
    """Pull (uri, user, password, database) from a Neo4jConfig across attr-name variants."""
    uri = (
        getattr(cfg, "uri", None)
        or getattr(cfg, "url", None)
        or "bolt://localhost:7687"
    )
    user = getattr(cfg, "user", None) or getattr(cfg, "username", None) or "neo4j"
    pwd = getattr(cfg, "password", None) or "password"
    db = getattr(cfg, "database", None) or getattr(cfg, "db_name", None) or "neo4j"
    return uri, user, pwd, db


class KB:
    """Thin read-only wrapper over a neo4j driver session."""

    def __init__(self, cfg: Neo4jConfig | None = None):
        from neo4j import GraphDatabase

        self.uri, self.user, self.pwd, self.db = _creds(cfg or Neo4jConfig())
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))

    def q(self, cypher: str, **params) -> list[dict]:
        """Run a read query, return a list of plain dicts."""
        with self._driver.session(database=self.db) as s:
            return [dict(r) for r in s.run(cypher, **params)]

    def scalar(self, cypher: str, **params):
        rows = self.q(cypher, **params)
        return next(iter(rows[0].values())) if rows else None

    def label_counts(self) -> dict[str, int]:
        return {
            r["l"]: r["n"]
            for r in self.q(
                "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS n ORDER BY n DESC"
            )
        }

    def rel_counts(self) -> dict[str, int]:
        return {
            r["t"]: r["n"]
            for r in self.q(
                "MATCH ()-[e]->() RETURN type(e) AS t, count(*) AS n ORDER BY n DESC"
            )
        }

    def nodes_by_label(self, label: str, props: list[str] | None = None) -> list[dict]:
        sel = "n" if not props else ", ".join(f"n.{p} AS {p}" for p in props)
        ret = "n" if not props else sel
        rows = self.q(f"MATCH (n:{label}) RETURN {ret}")
        return [r["n"] if not props else r for r in rows]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
