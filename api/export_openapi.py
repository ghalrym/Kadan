"""Write the generated OpenAPI specification to stdout."""

import json
import sys

from api.server import app


if __name__ == "__main__":
    json.dump(app.openapi(), sys.stdout, indent=2, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
