import unittest

from pydantic import ValidationError

from fastapi import HTTPException
from api.routes.v1.requests import RequestFilters, get_request, list_requests


class RequestFilteringTests(unittest.TestCase):
    def test_status_query_string_is_parsed(self):
        filters = RequestFilters.model_validate({"status": "200"})
        self.assertEqual(filters.status, 200)

    def test_unsupported_status_query_is_rejected(self):
        with self.assertRaises(ValidationError):
            RequestFilters.model_validate({"status": "201"})

    def test_filters_are_combined_before_pagination(self):
        result = list_requests(RequestFilters(type="LLM", status=200, limit=1, offset=1))
        self.assertEqual(result.total, 3)
        self.assertEqual([record.id for record in result.requests], ["req_002dc17b1707"])

    def test_search_is_case_insensitive(self):
        result = list_requests(RequestFilters(search="CUDA"))
        self.assertEqual([record.id for record in result.requests], ["req_3c916b21e808"])

    def test_missing_request_is_not_found(self):
        with self.assertRaises(HTTPException) as caught:
            get_request("missing")
        self.assertEqual(caught.exception.status_code, 404)
