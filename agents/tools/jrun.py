"""Run a shell command on the Jupyter server via a temporary kernel; prints its output."""
import json, os, sys, uuid, urllib.request
import truststore
truststore.inject_into_ssl()  # use the macOS trust store, as curl does
import websocket

BASE = "https://cis-neuron.mnsu.edu/user/15835731"
H = {"Authorization": f"token {os.environ['JUPYTERHUB_TOKEN']}"}


def api(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, headers={**H, "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=30) as r:
        txt = r.read()
        return json.loads(txt) if txt else None


cmd = sys.argv[1]
T = int(os.environ.get("JRUN_TIMEOUT", "120"))
code = f"import subprocess\nprint(subprocess.run({cmd!r}, shell=True, stdout=subprocess.PIPE, text=True, timeout={T}, stderr=subprocess.STDOUT).stdout)"
k = api("POST", "/api/kernels", {"name": "conda-base-py"})
try:
    ws = websocket.create_connection(BASE.replace("https", "wss") + f"/api/kernels/{k['id']}/channels",
                                     header=[f"{a}: {b}" for a, b in H.items()], timeout=T + 60)
    mid = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": mid, "msg_type": "execute_request", "username": "", "session": uuid.uuid4().hex, "version": "5.3"},
                        "parent_header": {}, "metadata": {}, "channel": "shell",
                        "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False}}))
    while True:
        m = json.loads(ws.recv())
        if m.get("parent_header", {}).get("msg_id") != mid:
            continue
        t = m["msg_type"]
        if t == "stream":
            sys.stdout.write(m["content"]["text"])
        elif t == "error":
            print("ERROR", m["content"]["ename"], m["content"]["evalue"])
        elif t == "status" and m["content"]["execution_state"] == "idle":
            break
    ws.close()
finally:
    api("DELETE", f"/api/kernels/{k['id']}")
