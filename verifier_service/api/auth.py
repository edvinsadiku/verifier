# api/auth.py
import time, hmac, hashlib
from django.http import JsonResponse
from django.conf import settings

ALLOWED_PATHS = {
    "/api/v1/health/",      # health pa auth
}

def verify_hmac_middleware(get_response):
    def middleware(request):
        # lejo health dhe OPTIONS pa auth
        if request.path in ALLOWED_PATHS or request.method == "OPTIONS":
            return get_response(request)

        secret = (settings.VERIFIER_SHARED_SECRET or "").encode()
        api_key = request.headers.get("X-API-Key")
        if api_key:
            if hmac.compare_digest(api_key, settings.VERIFIER_SHARED_SECRET):
                return get_response(request)
            return JsonResponse({"detail":"Invalid API key"}, status=401)

        ts = request.headers.get("X-Timestamp")
        sig = request.headers.get("X-Signature")
        if not ts or not sig:
            return JsonResponse({"detail":"Auth headers missing"}, status=401)
        try:
            ts = int(ts)
        except:
            return JsonResponse({"detail":"Bad timestamp"}, status=401)
        if abs(int(time.time()) - ts) > 300:
            return JsonResponse({"detail":"Stale timestamp"}, status=401)
        body = request.body or b""
        mac = hmac.new(secret, msg=(str(ts).encode()+b"."+body), digestmod=hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, sig):
            return JsonResponse({"detail":"Invalid signature"}, status=401)
        return get_response(request)
    return middleware
