import cv2
import glob
import argparse
from pathlib import Path

WINDOW_NAME = "Tennis Fixed Box Labeler"


class TennisFixedBoxLabeler:
    def __init__(self, image_dir, output_dir, class_id=0):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.class_id = class_id

        self.label_dir = self.output_dir / "labels"
        self.vis_dir = self.output_dir / "vis"
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = self._collect_images(self.image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        self.index = 0
        self.points = []
        self.boxes = []
        self.mode = "click"

        self.img_bgr = None
        self.display_bgr = None

        # Small red point
        self.point_radius = 2

        # Two fixed box sizes, both small
        self.small_half_size = 2   # box size = 7x7
        self.large_half_size = 4   # box size = 11x11

    def _collect_images(self, image_dir):
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(glob.glob(str(image_dir / ext)))
        return sorted(image_paths)

    def load_current_image(self):
        img_path = self.image_paths[self.index]
        self.img_bgr = cv2.imread(img_path)
        if self.img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        self.points = []
        self.boxes = []
        self.mode = "click"
        self.refresh_display()
        return img_path

    def refresh_display(self):
        out = self.img_bgr.copy()

        # draw clicked points
        for i, (x, y) in enumerate(self.points):
            cv2.circle(out, (x, y), self.point_radius, (0, 0, 255), -1)
            cv2.putText(
                out, str(i), (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1
            )

        # draw boxes in review mode
        if self.mode == "review":
            for i, (x1, y1, x2, y2) in enumerate(self.boxes):
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(
                    out, str(i), (x1, max(10, y1 - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1
                )

        info_lines = [
            f"Image {self.index + 1}/{len(self.image_paths)}: {Path(self.image_paths[self.index]).name}",
            "Left click: add red point",
            "d=generate boxes  s=save+next  u=undo  r=reset  n=next(no save)  q=quit",
            f"Mode: {self.mode.upper()} | points={len(self.points)} boxes={len(self.boxes)}",
        ]

        y0 = 22
        for line in info_lines:
            cv2.putText(out, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(out, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
            y0 += 24

        self.display_bgr = out

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode != "click":
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added point: {(x, y)}")
            self.refresh_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed_point = self.points.pop()
                print(f"Removed point: {removed_point}")
                self.refresh_display()
            else:
                print("No point to remove.")

    def make_fixed_box(self, x, y):
        H, W = self.img_bgr.shape[:2]

        # bottom 1/4 -> large box
        # from 1/2 to 3/4 -> small box
        # upper half -> small box
        if y >= H * 0.75:
            half_size = self.large_half_size
        else:
            half_size = self.small_half_size

        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(W - 1, x + half_size)
        y2 = min(H - 1, y + half_size)

        return (x1, y1, x2, y2)

    def detect_boxes_from_points(self):
        self.boxes = [self.make_fixed_box(x, y) for (x, y) in self.points]
        self.mode = "review"
        self.refresh_display()
        print(f"[Detect] points={len(self.points)} -> boxes={len(self.boxes)}")

    def undo_last_point(self):
        if self.mode == "review":
            self.mode = "click"
            self.boxes = []

        if self.points:
            self.points.pop()

        self.refresh_display()

    def reset_current(self):
        self.points = []
        self.boxes = []
        self.mode = "click"
        self.refresh_display()

    def xyxy_to_yolo(self, box):
        x1, y1, x2, y2 = box
        img_h, img_w = self.img_bgr.shape[:2]

        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0

        return xc / img_w, yc / img_h, bw / img_w, bh / img_h

    def save_current_labels(self):
        img_name = Path(self.image_paths[self.index]).name
        stem = Path(img_name).stem

        label_path = self.label_dir / f"{stem}.txt"
        lines = []

        for box in self.boxes:
            xc, yc, bw, bh = self.xyxy_to_yolo(box)
            lines.append(f"{self.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        vis_path = self.vis_dir / img_name
        cv2.imwrite(str(vis_path), self.display_bgr)

        print(f"[Saved] labels: {label_path}")
        print(f"[Saved] vis:    {vis_path}")
        print(f"[Saved] boxes:  {len(self.boxes)}")

    def next_image(self):
        self.index += 1
        if self.index >= len(self.image_paths):
            return False
        self.load_current_image()
        return True

    def run(self):
        img_path = self.load_current_image()
        print(f"Loaded: {img_path}")
        print("1) Left click to add red points.")
        print("2) Press 'd' to generate fixed boxes.")
        print("3) Press 's' to save and continue to next image.")
        print("4) Use 'u' undo, 'r' reset, 'n' next(no save), 'q' quit.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.display_bgr)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("Quit.")
                break

            elif key == ord('u'):
                self.undo_last_point()

            elif key == ord('r'):
                self.reset_current()

            elif key == ord('d'):
                self.detect_boxes_from_points()

            elif key == ord('s'):
                if self.mode != "review":
                    print("Please press 'd' first.")
                    continue

                self.save_current_labels()
                ok = self.next_image()
                if not ok:
                    print("All images finished.")
                    break
                print(f"Loaded: {self.image_paths[self.index]}")

            elif key == ord('n'):
                ok = self.next_image()
                if not ok:
                    print("All images finished.")
                    break
                print(f"Skipped. Loaded next: {self.image_paths[self.index]}")

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-size click labeler for tennis balls.")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing images.")
    parser.add_argument("--output_dir", type=str, default="./fixed_box_output", help="Folder to save labels and visualizations.")
    parser.add_argument("--class_id", type=int, default=0, help="YOLO class id.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = TennisFixedBoxLabeler(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        class_id=args.class_id
    )
    app.run()


if __name__ == "__main__":
    main()