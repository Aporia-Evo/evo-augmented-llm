from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class SpacetimeHttpClient:
    server_url: str
    database_name: str
    timeout_seconds: float = 10.0
    auth_token: str | None = None

    def _headers(self, *, json_body: bool = False, sql_body: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if json_body:
            headers["Content-Type"] = "application/json"
        if sql_body:
            headers["Content-Type"] = "text/plain"
        return headers

    def _base_path(self) -> str:
        database = quote(self.database_name, safe="")
        return f"{self.server_url.rstrip('/')}/v1/database/{database}"

    def call_reducer(self, reducer_name: str, *args: Any) -> requests.Response:
        reducer = quote(reducer_name, safe="")
        response = requests.post(
            f"{self._base_path()}/call/{reducer}",
            json=list(args),
            timeout=self.timeout_seconds,
            headers=self._headers(json_body=True),
        )
        response.raise_for_status()
        return response

    def sql(self, query: str) -> list[dict[str, Any]]:
        response = requests.post(
            f"{self._base_path()}/sql",
            data=query,
            timeout=self.timeout_seconds,
            headers=self._headers(sql_body=True),
        )
        response.raise_for_status()
        payload = response.json()
        rows: list[dict[str, Any]] = []
        for statement_result in payload:
            rows.extend(_parse_statement_rows(statement_result))
        return rows


def _parse_statement_rows(statement_result: dict[str, Any]) -> list[dict[str, Any]]:
    schema = statement_result.get("schema", {})
    row_values = statement_result.get("rows", [])
    elements = schema.get("elements", [])
    field_names: list[str] = []
    for index, element in enumerate(elements):
        name_payload = element.get("name", {})
        field_names.append(name_payload.get("some") or f"field_{index}")

    parsed_rows: list[dict[str, Any]] = []
    for row in row_values:
        if isinstance(row, dict):
            parsed_rows.append(row)
            continue
        if isinstance(row, list):
            parsed_rows.append(dict(zip(field_names, row, strict=False)))
            continue
        parsed_rows.append({"value": row})
    return parsed_rows

