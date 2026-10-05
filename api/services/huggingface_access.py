"""Read existing Hugging Face authorization for explicitly gated checkpoints."""
import os
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class OriginAuthorizationRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        """Never forward the Hub token to CDN hosts or an insecure redirect."""
        target = urlsplit(newurl)
        if target.scheme != 'https':
            raise ValueError('Refusing an insecure checkpoint redirect')
        redirected = super().redirect_request(request, response, code, message, headers, newurl)
        if redirected is not None and target.netloc != urlsplit(request.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


def open_gated_checkpoint(url, timeout=30):
    """Use an already configured token; never create credentials or accept Hub terms."""
    target = urlsplit(url)
    if target.scheme != 'https' or target.netloc != 'huggingface.co':
        raise ValueError('Gated checkpoint requests must originate at the Hugging Face Hub')
    token = os.environ.get('HF_TOKEN', '').strip()
    if not token:
        raise ValueError('This checkpoint requires approved Hugging Face access and an existing HF_TOKEN on the API server')
    request = Request(url, headers={'Authorization': f'Bearer {token}'})
    try:
        return build_opener(OriginAuthorizationRedirect()).open(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError('Hugging Face denied checkpoint access. Review the model terms on its official page and use an authorized HF_TOKEN.') from exc
        raise
