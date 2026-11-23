import binascii

from lxml import etree as ET
import base64
import io
from pathlib import Path
from PIL import Image
from SVGPatcher import SVGTemplate
import numpy as np

REPORT_TEMPLATE_PATH = (
    Path(__file__).resolve() / "../layouts/phantom_pack_report_template.svg"
).resolve()


def image_to_base64(image: str | Path | bytes | np.ndarray) -> str:
    if isinstance(image, (str, Path)):
        image = str(image)

        if image.startswith("data:image/"):
            return image

        try:
            data = Path(image).read_bytes()
        except OSError:
            try:
                data = base64.b64decode(image, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid image path or base64 string") from exc

    elif isinstance(image, bytes):
        data = image

    elif isinstance(image, np.ndarray):
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        data = buffer.getvalue()

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Normalize encoded images to PNG
    buffer = io.BytesIO()
    with Image.open(io.BytesIO(data)) as img:
        img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


TEMPLATE_PLACEHOLDERS = {
    "{{vial_header}}": "Measurements",
    "{{images_header_left}}": "PDFF Series Number:",
    "{{ihl_val}}": "13",
    "{{images_header_right}}": "FW Series Number:",
    "{{ihr_val}}": "1301",
    "{{accept}}": "Within Acceptance",
    "{{reject}}": "Outside Acceptance",
}


class Report:
    ACCEPT_COLOR = "#159C78"
    REJECT_COLOR = "#475569"
    CAUTION_COLOR = "#C9951C"

    def __init__(
        self,
        template: Path = REPORT_TEMPLATE_PATH,
        values: dict | None = None,
        images: list | None = None,
    ):
        self.placeholders = TEMPLATE_PLACEHOLDERS | (values or {})
        self.images = images or []
        self.tpl = SVGTemplate(template)

        self.measurements = None
        self.controls = [0, 10, 20, 30, 40]
        self.tolerance = 4
        self.pass_count = None
        self.result = None
        self._generated = False

    def set_images(self, images: list):
        self.images = images
        return self

    def set_measurements(
        self,
        measurements: list[float],
        controls: list[float] | None = None,
        tolerance: float = 4,
    ):
        if len(measurements) != 5:
            raise ValueError("Expected 5 measurements")

        controls = [0, 10, 20, 30, 40] if controls is None else controls

        if len(controls) != 5 or tolerance < 0:
            raise ValueError("Expected 5 controls and nonnegative tolerance")

        self.measurements = measurements
        self.controls = controls
        self.tolerance = tolerance

        passed = 0

        for i, (value, control) in enumerate(zip(measurements, controls)):
            low = control - tolerance
            high = control + tolerance
            accepted = low <= value <= high

            color = self.ACCEPT_COLOR if accepted else self.REJECT_COLOR
            self.set_vial_color(i, color)

            self.placeholders[f"{{{{v_{i}}}}}"] = f"{value:g}"
            self.placeholders[f"{{{{thresh_{i}}}}}"] = f"{low:g}% – {high:g}%"

            passed += accepted

        self.pass_count = passed

        if passed == 5:
            self.result = "PASS"
            color = self.ACCEPT_COLOR
        elif passed == 4:
            self.result = "BORDERLINE"
            color = self.CAUTION_COLOR
        else:
            self.result = "FAIL"
            color = self.REJECT_COLOR

        self.placeholders["{{pass}}"] = f"{passed}/5"
        self.placeholders["{{result}}"] = self.result
        self.set_result_color(color)
        return self

    def set_pdff_series(self, val: str | int):
        self.placeholders["{{ihl_val}}"] = str(val)

    def set_fw_series(self, val: str | int):
        self.placeholders["{{ihr_val}}"] = str(val)

    def set_vial_color(self, index: int, color: str):
        elem = self.tpl.get_elem(f"vile_color_{index}")
        if elem is not None:
            elem.set("stroke", color)
        return self

    def set_accept_color(self, color: str):
        self.ACCEPT_COLOR = color
        self.tpl.set_fill("accept_color", color)
        return self

    def set_reject_color(self, color: str):
        self.REJECT_COLOR = color
        self.tpl.set_fill("reject_color", color)
        return self

    def set_result_color(self, color: str):
        self.tpl.set_fill("result_text", color)
        self.tpl.set_fill("vials_passed", color)
        return self

    def generate(self, output_path: Path | str):
        for i in range(3):
            if i < len(self.images) and self.images[i] is not None:
                self.placeholders[f"img_{i}"] = image_to_base64(self.images[i])
            else:
                self.tpl.remove(f"img_{i}")

        self.tpl.replace(self.placeholders, new_copy=False)
        Path(output_path).write_bytes(self.tpl.png_bytes())
        return output_path
