"""Check the saved visualization in an isolated headless Chrome session.

Requires the optional `websocket-client` package and a local Chrome installation.
No external pages are loaded. Screenshots are written beside the HTML.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler

import websocket


def main():
    path = Path(sys.argv[1]).resolve()
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    with tempfile.TemporaryDirectory(prefix="heat-browser-") as profile:
        log = open(Path(profile) / "chrome.log", "w")
        process = subprocess.Popen([
            chrome, "--headless=new", "--no-first-run", "--no-default-browser-check",
            "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
            "--window-size=1500,1400", f"--user-data-dir={profile}",
            "--remote-debugging-port=0", "--remote-allow-origins=*", path.as_uri(),
        ], stdout=log, stderr=log)
        ws = None
        try:
            port_file = Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 25
            while not port_file.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError((Path(profile) / "chrome.log").read_text()[-3000:])
                time.sleep(0.1)
            port = port_file.read_text().splitlines()[0]
            opener = build_opener(ProxyHandler({}))
            targets = json.load(opener.open(f"http://127.0.0.1:{port}/json/list"))
            target = next(t for t in targets if t["type"] == "page")
            ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=45)
            request_id = 0

            def command(method, params):
                nonlocal request_id
                request_id += 1
                ws.send(json.dumps(dict(id=request_id, method=method, params=params)))
                while True:
                    response = json.loads(ws.recv())
                    if response.get("id") == request_id:
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        return response["result"]

            def evaluate(expression):
                result = command("Runtime.evaluate", dict(expression=expression, awaitPromise=True, returnByValue=True))
                if "exceptionDetails" in result:
                    raise RuntimeError(result["exceptionDetails"])
                return result["result"].get("value")

            while not evaluate("Boolean(document.querySelector('.js-plotly-plot')?._fullData?.length)"):
                if time.monotonic() > deadline:
                    raise RuntimeError("Plotly did not initialize")
                time.sleep(0.1)
            evaluate("window.gd=document.querySelector('.js-plotly-plot'); true")
            time.sleep(0.5)
            evaluate("""window.heatSnapshot = () => {
                const traces = gd.layout.meta?.error_traces || [2];
                const markers = gd.layout.meta?.error_markers || [8];
                const errors = traces.map(i=>Math.max(...gd._fullData[i].z.flat()));
                const markerErrors = markers.map(i=>gd.data[i].y[0]);
                const markerTimes = markers.map(i=>gd.data[i].x[0]);
                const images = Array.from(gd.querySelectorAll('.hm image'))
                    .map(image=>image.getAttribute('href') || image.getAttribute('xlink:href') || '');
                return {maxError:errors[0], markerTime:markerTimes[0], errors, markerErrors,
                    markerTimes, image:images.join('|'), imagePresent:images.length>=traces.length && images.every(Boolean)};
            }; true""")

            def snapshot(label):
                result = evaluate("heatSnapshot()")
                image = result.pop("image", None)
                result["imageHash"] = hashlib.sha256((image or "").encode()).hexdigest()[:16]
                result["imagePresent"] = result["imagePresent"] and bool(image)
                print(label, json.dumps(result))
                return result

            def select_step(index):
                evaluate("""(async () => {
                    const index = """ + str(index) + """;
                    const finished = new Promise((resolve, reject) => {
                        const timer = setTimeout(() => reject(new Error('Time step did not finish rendering')), 15000);
                        function done(event) {
                            if (event.detail.index !== index) return;
                            clearTimeout(timer);
                            gd.removeEventListener('heat-frame-rendered', done);
                            resolve();
                        }
                        gd.addEventListener('heat-frame-rendered', done);
                    });
                    gd.emit('plotly_sliderchange', {step:gd.layout.sliders[0].steps[index]});
                    await finished;
                })()""")

            def click_button(label):
                point = evaluate("""(() => {
                    const button = Array.from(gd.querySelectorAll('.updatemenu-button'))
                        .find(element => element.textContent.trim() === """ + json.dumps(label) + """);
                    if (!button) throw new Error('Button not found');
                    button.scrollIntoView({block:'center'});
                    const box = button.getBoundingClientRect();
                    return {x:box.x+box.width/2, y:box.y+box.height/2};
                })()""")
                command("Input.dispatchMouseEvent", dict(type="mousePressed", button="left", clickCount=1, **point))
                command("Input.dispatchMouseEvent", dict(type="mouseReleased", button="left", clickCount=1, **point))

            before = snapshot("initial")
            select_step(0)
            zero = snapshot("t=0")
            last = evaluate("gd.layout.sliders[0].steps.length-1")
            select_step(last // 4)
            middle = snapshot("t=tf/4")
            select_step(last)
            end = snapshot("t=tf")
            # After a pulse the final field can be below the color scale's
            # precision, legitimately giving the same pixels as the zero field.
            if not (zero["maxError"] == 0 and zero["imagePresent"] and zero["imageHash"] != before["imageHash"]
                    and middle["imageHash"] != zero["imageHash"] and end["markerTime"] > middle["markerTime"]):
                raise AssertionError("Heatmap data or rendered image did not update with the time step")
            print("Browser: heatmap changes with the time step")
            evaluate("""(() => {
                window.heatSamples = [];
                gd.addEventListener('heat-frame-rendered', event => {
                    const state = heatSnapshot();
                    const href = state.image;
                    let hash = 0; for (let i=0;i<href.length;i++) hash = ((hash<<5)-hash+href.charCodeAt(i))|0;
                    heatSamples.push({t:event.detail.time, markerTime:state.markerTime,
                        maxError:state.maxError, markerError:state.markerErrors[0],
                        synchronized:state.errors.every((e,i)=>e===state.markerErrors[i] && state.markerTimes[i]===event.detail.time),
                        imagePresent:state.imagePresent, hash});
                });
                return true;
            })()""")
            click_button("Abspielen")
            evaluate("""new Promise((resolve, reject) => {
                const deadline=Date.now()+15000;
                const timer=setInterval(() => {
                    if (heatSamples.length>=4) {clearInterval(timer); resolve(true);}
                    else if (Date.now()>deadline) {clearInterval(timer); reject(new Error('Playback did not advance'));}
                }, 50);
            })""")
            click_button("Pause")
            time.sleep(0.5)
            samples = evaluate("heatSamples")
            print("playback", json.dumps(samples))
            assert len(samples) >= 4
            assert all(s["imagePresent"] and s["synchronized"] for s in samples)
            assert len({s["hash"] for s in samples}) >= 4
            time.sleep(0.3)
            assert evaluate("heatSamples.length") == len(samples), "Pause did not stop playback"
            print("Browser: Play/Pause redraw the heatmap in sync with the time markers")
            import base64
            screenshot = command("Page.captureScreenshot", dict(format="png"))["data"]
            path.with_suffix(".browser.png").write_bytes(base64.b64decode(screenshot))
        finally:
            if ws is not None:
                ws.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
