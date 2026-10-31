"""Configure the pinned Openbox image without writing into the persistent home."""

import xml.etree.ElementTree as ET
from pathlib import Path

NS = "http://openbox.org/3.4/rc"
ET.register_namespace("", NS)


def configure_openbox(source: str) -> str:
    root = ET.fromstring(source)

    def setting(path: str, value: str) -> None:
        root.find(path, {"o": NS}).text = value

    setting("o:desktops/o:number", "1")
    setting("o:theme/o:name", "Breeze-ob")
    setting("o:theme/o:titleLayout", "NLIMC")
    for font in root.findall("o:theme/o:font/o:size", {"o": NS}):
        font.text = "10"
    menu = root.find("o:menu", {"o": NS})
    for file in menu.findall("o:file", {"o": NS}):
        menu.remove(file)
    ET.SubElement(menu, f"{{{NS}}}file").text = "/opt/kadan/desktop/menu.xml"
    keyboard = root.find("o:keyboard", {"o": NS})
    for binding in keyboard.findall("o:keybind", {"o": NS}):
        if binding.get("key") == "W-e":
            keyboard.remove(binding)
    for key, command in (("W-b", "browser"), ("C-A-t", "terminal")):
        binding = ET.SubElement(keyboard, f"{{{NS}}}keybind", key=key)
        action = ET.SubElement(binding, f"{{{NS}}}action", name="Execute")
        ET.SubElement(
            action, f"{{{NS}}}command"
        ).text = f"python3 /opt/kadan/desktop/launch.py {command}"
    binding = ET.SubElement(keyboard, f"{{{NS}}}keybind", key="C-Escape")
    action = ET.SubElement(binding, f"{{{NS}}}action", name="ShowMenu")
    ET.SubElement(action, f"{{{NS}}}menu").text = "root-menu"
    applications = root.find("o:applications", {"o": NS})
    browser = ET.SubElement(
        applications, f"{{{NS}}}application", {"class": "*hrom*", "role": "browser"}
    )
    ET.SubElement(browser, f"{{{NS}}}decor").text = "yes"
    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    Path("/opt/gem/openbox.sh").write_text(
        "#!/bin/sh\n"
        "exec /usr/bin/openbox --config-file /opt/gem/openbox.xml "
        "--startup 'xsetroot -solid \"#0e0f12\"'\n"
    )
    Path("/opt/gem/openbox.xml").write_text(
        configure_openbox(Path("/etc/xdg/openbox/rc.xml").read_text())
    )
    Path("/opt/gem/supervisord/kadan-desktop.conf").write_text(
        Path("/opt/kadan/desktop/panel.conf").read_text()
    )
