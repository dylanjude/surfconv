#!/usr/bin/env python3
"""
SurfConv TUI - Interactive Terminal Dashboard for Surface Mesh Conversion

Provides file browsing, mesh statistics, topology checking, format selection,
and conversion in a single curses-based terminal interface.

Usage:
    python surfconv_tui.py [directory]
"""

import curses
import os
import sys
import io
from contextlib import redirect_stdout

from surfconv import SurfaceMesh, read_mesh, write_mesh, detect_format, FORMATS
from mesh_check import check_mesh_topology, check_euler_characteristic

# Supported file extensions for the file browser
SUPPORTED_EXTS = {'.ugrid', '.vtk', '.vtu', '.stl', '.facet', '.dat'}

# Output formats the user can cycle through (exclude stl-binary as a separate pick;
# we just use 'stl' which writes ASCII by default)
OUTPUT_FORMATS = [k for k in FORMATS if k != 'stl-binary']


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def capture_output(func, *args, **kwargs):
    """Run *func* while capturing stdout. Returns (result, captured_text)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def fmt_number(n):
    """Format an integer with comma separators."""
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Color pairs (initialised in App.__init__)
# ---------------------------------------------------------------------------
CP_BORDER       = 1
CP_FOCUS_BORDER = 2
CP_DIR_ENTRY    = 3
CP_CURSOR       = 4
CP_SUCCESS      = 5
CP_ERROR        = 6
CP_WARNING      = 7
CP_TITLE        = 8
CP_STATUS       = 9
CP_FMT_SEL      = 10


# ---------------------------------------------------------------------------
# Panel base class
# ---------------------------------------------------------------------------

class Panel:
    """Base class for a bordered panel in the TUI."""

    def __init__(self, title=""):
        self.title = title
        self.y = self.x = self.h = self.w = 0
        self.focused = False
        self.scroll_offset = 0

    def resize(self, y, x, h, w):
        self.y, self.x, self.h, self.w = y, x, h, w

    @property
    def inner_h(self):
        return max(self.h - 2, 0)

    @property
    def inner_w(self):
        return max(self.w - 2, 0)

    def draw_border(self, stdscr):
        if self.h < 2 or self.w < 2:
            return
        cp = CP_FOCUS_BORDER if self.focused else CP_BORDER
        attr = curses.color_pair(cp)
        # Draw box using unicode-safe characters
        try:
            stdscr.attron(attr)
            # top
            stdscr.addch(self.y, self.x, curses.ACS_ULCORNER)
            stdscr.hline(self.y, self.x + 1, curses.ACS_HLINE, self.w - 2)
            stdscr.addch(self.y, self.x + self.w - 1, curses.ACS_URCORNER)
            # sides
            for row in range(1, self.h - 1):
                stdscr.addch(self.y + row, self.x, curses.ACS_VLINE)
                stdscr.addch(self.y + row, self.x + self.w - 1, curses.ACS_VLINE)
            # bottom
            stdscr.addch(self.y + self.h - 1, self.x, curses.ACS_LLCORNER)
            stdscr.hline(self.y + self.h - 1, self.x + 1, curses.ACS_HLINE, self.w - 2)
            # Avoid writing to bottom-right corner (curses quirk)
            try:
                stdscr.addch(self.y + self.h - 1, self.x + self.w - 1, curses.ACS_LRCORNER)
            except curses.error:
                pass
            stdscr.attroff(attr)
        except curses.error:
            pass

        # Title
        if self.title:
            label = f" {self.title} "
            tx = self.x + 2
            try:
                stdscr.addnstr(self.y, tx, label,
                               min(len(label), self.w - 4),
                               curses.color_pair(CP_FOCUS_BORDER) | curses.A_BOLD
                               if self.focused else attr | curses.A_BOLD)
            except curses.error:
                pass

    def safe_addnstr(self, stdscr, row, col, text, maxlen, attr=0):
        """Write text clamped to panel interior, ignoring curses boundary errors."""
        if row < 0 or col < 0 or maxlen <= 0:
            return
        try:
            stdscr.addnstr(row, col, text, maxlen, attr)
        except curses.error:
            pass

    def draw(self, stdscr):
        self.draw_border(stdscr)

    def handle_key(self, key):
        pass


# ---------------------------------------------------------------------------
# File Browser Panel
# ---------------------------------------------------------------------------

class FileBrowserPanel(Panel):
    def __init__(self, start_dir="."):
        super().__init__("FILES")
        self.cwd = os.path.abspath(start_dir)
        self.entries = []       # list of (name, is_dir)
        self.cursor = 0
        self.filter_mode = False
        self.filter_text = ""
        self.refresh_listing()

    def refresh_listing(self):
        self.entries = []
        self.entries.append(("..", True))
        try:
            items = sorted(os.listdir(self.cwd), key=lambda s: s.lower())
        except PermissionError:
            items = []
        for name in items:
            full = os.path.join(self.cwd, name)
            if os.path.isdir(full):
                self.entries.append((name, True))
            else:
                ext = os.path.splitext(name)[1].lower()
                if ext in SUPPORTED_EXTS:
                    self.entries.append((name, False))
        if self.filter_mode and self.filter_text:
            ft = self.filter_text.lower()
            self.entries = [(n, d) for n, d in self.entries
                           if d or ft in n.lower()]
        self.cursor = min(self.cursor, max(len(self.entries) - 1, 0))

    def selected_path(self):
        if not self.entries:
            return None
        name, is_dir = self.entries[self.cursor]
        return os.path.join(self.cwd, name)

    def selected_is_dir(self):
        if not self.entries:
            return False
        return self.entries[self.cursor][1]

    def enter_selected(self):
        """Navigate into directory or return selected file path."""
        if not self.entries:
            return None
        name, is_dir = self.entries[self.cursor]
        if is_dir:
            if name == "..":
                self.cwd = os.path.dirname(self.cwd)
            else:
                self.cwd = os.path.join(self.cwd, name)
            self.cursor = 0
            self.scroll_offset = 0
            self.filter_mode = False
            self.filter_text = ""
            self.refresh_listing()
            return None
        return os.path.join(self.cwd, name)

    def go_parent(self):
        self.cwd = os.path.dirname(self.cwd)
        self.cursor = 0
        self.scroll_offset = 0
        self.filter_mode = False
        self.filter_text = ""
        self.refresh_listing()

    def draw(self, stdscr):
        self.draw_border(stdscr)
        iw = self.inner_w
        ih = self.inner_h
        if iw <= 0 or ih <= 0:
            return

        # Adjust scroll
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        if self.cursor >= self.scroll_offset + ih:
            self.scroll_offset = self.cursor - ih + 1

        row0 = self.y + 1
        col0 = self.x + 1
        for i in range(ih):
            idx = self.scroll_offset + i
            if idx >= len(self.entries):
                break
            name, is_dir = self.entries[idx]
            display = name + "/" if is_dir else name
            if idx == self.cursor:
                attr = curses.color_pair(CP_CURSOR)
                prefix = "> "
            elif is_dir:
                attr = curses.color_pair(CP_DIR_ENTRY)
                prefix = "  "
            else:
                attr = 0
                prefix = "  "
            line = prefix + display
            self.safe_addnstr(stdscr, row0 + i, col0, line.ljust(iw), iw, attr)

        # Filter indicator
        if self.filter_mode:
            ftxt = f"/{self.filter_text}"
            self.safe_addnstr(stdscr, self.y + self.h - 1, self.x + 2,
                              f" {ftxt} ", min(len(ftxt) + 2, self.w - 4),
                              curses.color_pair(CP_WARNING))

    def handle_key(self, key):
        if self.filter_mode:
            if key in (27,):  # ESC
                self.filter_mode = False
                self.filter_text = ""
                self.refresh_listing()
            elif key in (curses.KEY_BACKSPACE, 127, 263):
                self.filter_text = self.filter_text[:-1]
                self.refresh_listing()
            elif key in (10, curses.KEY_ENTER):
                self.filter_mode = False
            elif 32 <= key < 127:
                self.filter_text += chr(key)
                self.refresh_listing()
            return None

        if key == ord('/'):
            self.filter_mode = True
            self.filter_text = ""
            return None
        if key in (ord('j'), curses.KEY_DOWN):
            self.cursor = min(self.cursor + 1, len(self.entries) - 1)
        elif key in (ord('k'), curses.KEY_UP):
            self.cursor = max(self.cursor - 1, 0)
        elif key in (10, curses.KEY_ENTER):
            return self.enter_selected()
        elif key in (curses.KEY_BACKSPACE, 127, 263):
            self.go_parent()
        return None


# ---------------------------------------------------------------------------
# Mesh Info Panel
# ---------------------------------------------------------------------------

class MeshInfoPanel(Panel):
    def __init__(self):
        super().__init__("MESH INFO")
        self.lines = []

    def set_mesh(self, mesh, filepath):
        name = os.path.basename(filepath)
        fmt = detect_format(filepath) or "?"
        verts = mesh.vertices
        bb_min = verts.min(axis=0) if len(verts) else [0, 0, 0]
        bb_max = verts.max(axis=0) if len(verts) else [0, 0, 0]
        self.lines = [
            f"File:      {name}",
            f"Format:    {fmt.upper()}",
            f"Vertices:  {fmt_number(len(mesh.vertices))}",
            f"Triangles: {fmt_number(len(mesh.triangles))}",
            f"Quads:     {fmt_number(len(mesh.quads))}",
            "",
            "Bounding Box:",
            f"  X: [{bb_min[0]:.4f}, {bb_max[0]:.4f}]",
            f"  Y: [{bb_min[1]:.4f}, {bb_max[1]:.4f}]",
            f"  Z: [{bb_min[2]:.4f}, {bb_max[2]:.4f}]",
        ]

    def clear(self):
        self.lines = ["(no mesh loaded)"]

    def draw(self, stdscr):
        self.draw_border(stdscr)
        iw = self.inner_w
        ih = self.inner_h
        if iw <= 0 or ih <= 0:
            return
        row0 = self.y + 1
        col0 = self.x + 1
        for i, line in enumerate(self.lines[:ih]):
            self.safe_addnstr(stdscr, row0 + i, col0, line, iw)


# ---------------------------------------------------------------------------
# Topology Panel
# ---------------------------------------------------------------------------

class TopologyPanel(Panel):
    def __init__(self):
        super().__init__("TOPOLOGY")
        self.lines = []
        self.attrs = []  # per-line color pair or 0

    def set_result(self, result, chi):
        self.lines = []
        self.attrs = []

        def add(text, cp=0):
            self.lines.append(text)
            self.attrs.append(cp)

        add(f"Edges: {fmt_number(result['n_edges'])}")
        add(f"Interior:     {fmt_number(result['n_interior_edges'])}")
        add(f"Boundary:     {fmt_number(result['n_boundary_edges'])}")

        nm = result['n_nonmanifold_edges']
        add(f"Non-manifold: {fmt_number(nm)}",
            CP_ERROR if nm > 0 else CP_SUCCESS)

        add("")
        add(f"Euler char (V-E+F): {chi}")

        if result['n_boundary_edges'] == 0:
            if chi == 2:
                add("  Sphere-like closed surface", CP_SUCCESS)
            elif chi == 0:
                add("  Torus-like closed surface")
            else:
                add(f"  Genus ~{(2 - chi) // 2}")
        else:
            add(f"  Open surface ({result['n_boundary_edges']} boundary edges)")

        add("")
        inc = result['n_inconsistent_edges']
        if inc == 0:
            add("Orientation: CONSISTENT", CP_SUCCESS)
        else:
            add(f"Orientation: INCONSISTENT ({inc} edges)", CP_ERROR)

        if result['consistent']:
            add("Result: PASS", CP_SUCCESS)
        else:
            add("Result: FAIL", CP_ERROR)

    def clear(self):
        self.lines = ["(press 't' to run check)"]
        self.attrs = [0]

    def draw(self, stdscr):
        self.draw_border(stdscr)
        iw = self.inner_w
        ih = self.inner_h
        if iw <= 0 or ih <= 0:
            return
        row0 = self.y + 1
        col0 = self.x + 1
        for i in range(min(len(self.lines), ih)):
            attr = curses.color_pair(self.attrs[i]) if i < len(self.attrs) and self.attrs[i] else 0
            self.safe_addnstr(stdscr, row0 + i, col0, self.lines[i], iw, attr)


# ---------------------------------------------------------------------------
# Output / Log Panel
# ---------------------------------------------------------------------------

class OutputLogPanel(Panel):
    def __init__(self):
        super().__init__("OUTPUT / LOG")
        self.format_index = 0
        self.output_file = ""
        self.log_lines = []
        self.scroll_offset = 0

    def set_default_format(self, input_path):
        """Pick the first format that differs from the input format."""
        in_fmt = detect_format(input_path)
        for i, fmt in enumerate(OUTPUT_FORMATS):
            if fmt != in_fmt:
                self.format_index = i
                break
        self._update_output_file(input_path)

    def _update_output_file(self, input_path=None):
        if input_path is None:
            return
        self._input_path = input_path
        base = os.path.splitext(input_path)[0]
        fmt = OUTPUT_FORMATS[self.format_index]
        ext = FORMATS[fmt]['ext']
        self.output_file = base + ext

    def cycle_format(self, delta):
        self.format_index = (self.format_index + delta) % len(OUTPUT_FORMATS)
        if hasattr(self, '_input_path'):
            self._update_output_file(self._input_path)

    def add_log(self, text, error=False):
        for line in text.splitlines():
            self.log_lines.append((line, error))
        # keep a reasonable buffer
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[-500:]
        # auto-scroll to bottom
        self.scroll_offset = max(len(self.log_lines) - self._log_visible(), 0)

    def _log_visible(self):
        return max(self.inner_h - 3, 1)

    def selected_format(self):
        return OUTPUT_FORMATS[self.format_index]

    def draw(self, stdscr):
        self.draw_border(stdscr)
        iw = self.inner_w
        ih = self.inner_h
        if iw <= 0 or ih <= 0:
            return
        row0 = self.y + 1
        col0 = self.x + 1

        # Format selector row
        parts = []
        for i, fmt in enumerate(OUTPUT_FORMATS):
            if i == self.format_index:
                parts.append(f"[{fmt}]")
            else:
                parts.append(f" {fmt} ")
        fmt_line = "Format: " + " ".join(parts)
        # Draw format line with highlight on selected
        self.safe_addnstr(stdscr, row0, col0, "Format: ", iw)
        cx = col0 + 8
        for i, fmt in enumerate(OUTPUT_FORMATS):
            if cx >= col0 + iw:
                break
            if i == self.format_index:
                token = f"[{fmt}]"
                attr = curses.color_pair(CP_FMT_SEL) | curses.A_BOLD
            else:
                token = f" {fmt} "
                attr = 0
            self.safe_addnstr(stdscr, row0, cx, token, col0 + iw - cx, attr)
            cx += len(token)

        # Output file row
        if self.output_file:
            oname = os.path.basename(self.output_file)
            self.safe_addnstr(stdscr, row0 + 1, col0,
                              f"Output: {oname}", iw)

        # Separator
        sep_row = row0 + 2
        if sep_row < self.y + self.h - 1:
            self.safe_addnstr(stdscr, sep_row, col0, "-" * iw, iw,
                              curses.color_pair(CP_BORDER))

        # Log lines
        log_start = row0 + 3
        visible = self._log_visible()
        for i in range(visible):
            li = self.scroll_offset + i
            if li >= len(self.log_lines):
                break
            text, is_err = self.log_lines[li]
            attr = curses.color_pair(CP_ERROR) if is_err else 0
            self.safe_addnstr(stdscr, log_start + i, col0, "> " + text,
                              iw, attr)

    def handle_key(self, key):
        if key in (ord('h'), curses.KEY_LEFT):
            self.cycle_format(-1)
        elif key in (ord('l'), curses.KEY_RIGHT):
            self.cycle_format(1)
        elif key in (ord('j'), curses.KEY_DOWN):
            max_off = max(len(self.log_lines) - self._log_visible(), 0)
            self.scroll_offset = min(self.scroll_offset + 1, max_off)
        elif key in (ord('k'), curses.KEY_UP):
            self.scroll_offset = max(self.scroll_offset - 1, 0)


# ---------------------------------------------------------------------------
# Help Overlay
# ---------------------------------------------------------------------------

class HelpOverlay:
    HELP_LINES = [
        "SurfConv TUI - Key Bindings",
        "",
        "  Tab          Cycle focus between panels",
        "  j / Down     Navigate down",
        "  k / Up       Navigate up",
        "  Enter        Open directory / load mesh file",
        "  Backspace    Go to parent directory",
        "  h / Left     Previous output format",
        "  l / Right    Next output format",
        "  /            Filter files in browser",
        "  c            Start conversion",
        "  t            Run topology check",
        "  q            Quit",
        "  ?            Toggle this help",
        "",
        "  Press any key to close",
    ]

    def draw(self, stdscr, max_y, max_x):
        h = len(self.HELP_LINES) + 2
        w = max(len(l) for l in self.HELP_LINES) + 4
        # Center
        y0 = max((max_y - h) // 2, 0)
        x0 = max((max_x - w) // 2, 0)
        h = min(h, max_y - y0)
        w = min(w, max_x - x0)

        attr = curses.color_pair(CP_TITLE) | curses.A_BOLD
        for row in range(h):
            try:
                stdscr.addnstr(y0 + row, x0, " " * w, w, attr)
            except curses.error:
                pass
        for i, line in enumerate(self.HELP_LINES):
            if i + 1 >= h - 1:
                break
            try:
                stdscr.addnstr(y0 + 1 + i, x0 + 2, line, w - 4, attr)
            except curses.error:
                pass


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class App:
    MIN_COLS = 60
    MIN_ROWS = 20

    def __init__(self, stdscr, start_dir="."):
        self.stdscr = stdscr
        curses.curs_set(0)
        stdscr.timeout(-1)

        # Colours
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(CP_BORDER,       curses.COLOR_WHITE,  -1)
        curses.init_pair(CP_FOCUS_BORDER, curses.COLOR_CYAN,   -1)
        curses.init_pair(CP_DIR_ENTRY,    curses.COLOR_BLUE,   -1)
        curses.init_pair(CP_CURSOR,       curses.COLOR_BLACK,  curses.COLOR_WHITE)
        curses.init_pair(CP_SUCCESS,      curses.COLOR_GREEN,  -1)
        curses.init_pair(CP_ERROR,        curses.COLOR_RED,    -1)
        curses.init_pair(CP_WARNING,      curses.COLOR_YELLOW, -1)
        curses.init_pair(CP_TITLE,        curses.COLOR_BLACK,  curses.COLOR_CYAN)
        curses.init_pair(CP_STATUS,       curses.COLOR_BLACK,  curses.COLOR_WHITE)
        curses.init_pair(CP_FMT_SEL,      curses.COLOR_BLACK,  curses.COLOR_GREEN)

        # Panels
        self.file_panel = FileBrowserPanel(start_dir)
        self.info_panel = MeshInfoPanel()
        self.topo_panel = TopologyPanel()
        self.out_panel  = OutputLogPanel()

        self.panels = [self.file_panel, self.info_panel,
                       self.topo_panel, self.out_panel]
        self.focus_idx = 0
        self.panels[0].focused = True

        self.info_panel.clear()
        self.topo_panel.clear()

        self.help_overlay = HelpOverlay()
        self.show_help = False

        # Mesh state
        self.mesh = None
        self.mesh_path = None

        # Threading
        self._busy = False
        self._busy_label = ""

    # -- layout --------------------------------------------------------------

    def _compute_layout(self):
        max_y, max_x = self.stdscr.getmaxyx()
        self._max_y = max_y
        self._max_x = max_x

        if max_y < self.MIN_ROWS or max_x < self.MIN_COLS:
            return False

        # Status bar takes the last row
        usable_h = max_y - 1

        # Left column: 1/3 width, min 28
        left_w = max(max_x // 3, 28)
        left_w = min(left_w, max_x - 32)  # leave room for right
        right_w = max_x - left_w

        # File panel fills left column
        self.file_panel.resize(0, 0, usable_h, left_w)

        # Right column: three panels stacked
        third = usable_h // 3
        remainder = usable_h - 3 * third

        info_h = third
        topo_h = third
        out_h = third + remainder  # give extra rows to log

        self.info_panel.resize(0, left_w, info_h, right_w)
        self.topo_panel.resize(info_h, left_w, topo_h, right_w)
        self.out_panel.resize(info_h + topo_h, left_w, out_h, right_w)

        return True

    # -- drawing -------------------------------------------------------------

    def _draw_status_bar(self):
        max_y, max_x = self._max_y, self._max_x
        bar_y = max_y - 1
        attr = curses.color_pair(CP_STATUS)
        line = " Tab:panel  j/k:nav  Enter:select  c:convert  t:check  q:quit  ?:help "
        if self._busy:
            line = f" {self._busy_label}... " + line
        try:
            self.stdscr.addnstr(bar_y, 0, line.ljust(max_x), max_x, attr)
        except curses.error:
            pass

    def _draw_title_bar(self):
        """Draw a small title inset into the file panel top border."""
        pass  # Titles are drawn by Panel.draw_border

    def _draw(self):
        self.stdscr.erase()

        if not self._compute_layout():
            # Terminal too small
            msg = "Please resize terminal (min 60x20)"
            try:
                self.stdscr.addnstr(self._max_y // 2,
                                    max((self._max_x - len(msg)) // 2, 0),
                                    msg, self._max_x,
                                    curses.color_pair(CP_WARNING) | curses.A_BOLD)
            except curses.error:
                pass
            self.stdscr.refresh()
            return

        for p in self.panels:
            p.draw(self.stdscr)

        self._draw_status_bar()

        if self.show_help:
            self.help_overlay.draw(self.stdscr, self._max_y, self._max_x)

        self.stdscr.refresh()

    # -- mesh operations (threaded) ------------------------------------------

    def _show_busy(self, label):
        """Display a busy indicator and immediately refresh the screen."""
        self._busy = True
        self._busy_label = label
        self._draw()

    def load_mesh(self, path):
        """Load a mesh file (synchronous — GIL-bound, threading won't help)."""
        self.out_panel.add_log(f"Loading {os.path.basename(path)}...")
        self._show_busy("Loading mesh")
        try:
            mesh, output = capture_output(read_mesh, path)
            self.mesh = mesh
            self.mesh_path = path
            self.info_panel.set_mesh(mesh, path)
            self.topo_panel.clear()
            self.out_panel.set_default_format(path)
            if output.strip():
                self.out_panel.add_log(output.strip())
        except Exception as e:
            self.out_panel.add_log(f"Error: {e}", error=True)
            self.mesh = None
            self.mesh_path = None
            self.info_panel.clear()
            self.topo_panel.clear()
        finally:
            self._busy = False

    def run_topology_check(self):
        if self.mesh is None:
            self.out_panel.add_log("No mesh loaded", error=True)
            return

        self.out_panel.add_log("Running topology check...")
        self._show_busy("Checking topology")
        try:
            result = check_mesh_topology(self.mesh)
            chi = check_euler_characteristic(self.mesh, result)
            self.topo_panel.set_result(result, chi)
            status = "PASS" if result['consistent'] else "FAIL"
            self.out_panel.add_log(f"Topology check: {status}")
        except Exception as e:
            self.out_panel.add_log(f"Topology error: {e}", error=True)
            self.topo_panel.clear()
        finally:
            self._busy = False

    def start_conversion(self):
        if self.mesh is None:
            self.out_panel.add_log("No mesh loaded", error=True)
            return

        out_fmt = self.out_panel.selected_format()
        out_file = self.out_panel.output_file

        self.out_panel.add_log(f"Converting to {out_fmt.upper()} -> {os.path.basename(out_file)}")
        self._show_busy("Converting")
        try:
            _, output = capture_output(write_mesh, self.mesh, out_file, out_fmt)
            if output.strip():
                self.out_panel.add_log(output.strip())
            self.out_panel.add_log("Conversion complete!")
            self.file_panel.refresh_listing()
        except Exception as e:
            self.out_panel.add_log(f"Conversion error: {e}", error=True)
        finally:
            self._busy = False

    # -- focus ---------------------------------------------------------------

    def _cycle_focus(self):
        self.panels[self.focus_idx].focused = False
        self.focus_idx = (self.focus_idx + 1) % len(self.panels)
        self.panels[self.focus_idx].focused = True

    # -- main loop -----------------------------------------------------------

    def run(self):
        while True:
            self._draw()
            try:
                key = self.stdscr.getch()
            except curses.error:
                continue

            if key == curses.KEY_RESIZE:
                continue

            if self.show_help:
                self.show_help = False
                continue

            if key == ord('q'):
                break
            elif key == ord('?'):
                self.show_help = True
            elif key == 9:  # Tab
                self._cycle_focus()
            elif key == ord('c'):
                if not self._busy:
                    self.start_conversion()
            elif key == ord('t'):
                if not self._busy:
                    self.run_topology_check()
            else:
                # Delegate to focused panel
                result = self.panels[self.focus_idx].handle_key(key)
                # FileBrowserPanel returns a file path on Enter
                if result is not None and self.focus_idx == 0:
                    if not self._busy:
                        self.load_mesh(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    start_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    curses.wrapper(lambda stdscr: App(stdscr, start_dir).run())


if __name__ == "__main__":
    main()
