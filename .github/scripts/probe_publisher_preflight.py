"""One-shot, unauthenticated public metadata probe; never publishes or reads secrets."""
import urllib.error
import urllib.request

from publish_pr import ENVIRONMENT, ROOT, NoRedirect, http_diagnostic


def main():
    """GET each fixed preflight endpoint once and print only safe status/rate headers."""
    opener = urllib.request.build_opener(NoRedirect())
    paths = (ROOT, ROOT + '/environments/' + ENVIRONMENT,
             ROOT + '/environments/' + ENVIRONMENT +
             '/deployment-branch-policies?per_page=100&page=1')
    for path in paths:
        request = urllib.request.Request('https://api.github.com' + path, headers={
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'Kadan-PR-publisher'})
        try:
            with opener.open(request, timeout=30) as response:
                print(http_diagnostic(path, response.status, response.headers), flush=True)
        except urllib.error.HTTPError as exc:
            print(http_diagnostic(path, exc.code, exc.headers), flush=True)
            exc.close()
        except (urllib.error.URLError, OSError):
            print('Transport failure; no HTTP response available', flush=True)


if __name__ == '__main__':
    main()
