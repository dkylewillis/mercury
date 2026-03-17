"""
Mercury GUI — tkinter desktop interface for document management and Q&A.

Provides:
  - Connect to a collection (data directory + collection name)
  - List / filter / delete documents
  - Ingest PDFs (with chunked conversion support)
  - Ask questions with markdown-rendered answers and source citations
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from agents import expand_query, answer_question
from chunker import Chunker
from converter import Converter
from document_store import DocumentStore
from vector_store import VectorStore

DEFAULT_DATA_DIR = os.environ.get("MERCURY_DATA_DIR", "./mercury_data")
DEFAULT_COLLECTION = "mercury"


# ---------------------------------------------------------------------------
# Markdown renderer for tk.Text
# ---------------------------------------------------------------------------

def render_markdown(text_widget: tk.Text, md: str) -> None:
    """Parse markdown string and insert formatted text into a tk.Text widget."""
    text_widget.configure(state="normal")
    text_widget.delete("1.0", "end")

    # Define tags
    base_font = ("Segoe UI", 11)
    text_widget.configure(font=base_font, wrap="word", padx=12, pady=8)
    text_widget.tag_configure("h1", font=("Segoe UI", 18, "bold"), spacing3=6)
    text_widget.tag_configure("h2", font=("Segoe UI", 15, "bold"), spacing3=4)
    text_widget.tag_configure("h3", font=("Segoe UI", 13, "bold"), spacing3=3)
    text_widget.tag_configure("bold", font=("Segoe UI", 11, "bold"))
    text_widget.tag_configure("italic", font=("Segoe UI", 11, "italic"))
    text_widget.tag_configure("bold_italic", font=("Segoe UI", 11, "bold italic"))
    text_widget.tag_configure("code", font=("Consolas", 10), background="#e8e8e8",
                              relief="flat", borderwidth=1)
    text_widget.tag_configure("code_block", font=("Consolas", 10), background="#f0f0f0",
                              lmargin1=20, lmargin2=20, rmargin=20, spacing1=4, spacing3=4)
    text_widget.tag_configure("citation", foreground="#0066cc", font=("Segoe UI", 11, "bold"))
    text_widget.tag_configure("bullet", lmargin1=20, lmargin2=34)
    text_widget.tag_configure("blockquote", lmargin1=20, lmargin2=20,
                              foreground="#555555", font=("Segoe UI", 11, "italic"))

    lines = md.split("\n")
    in_code_block = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code block toggle
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                pass  # opening — skip this line
            i += 1
            continue

        if in_code_block:
            text_widget.insert("end", line + "\n", "code_block")
            i += 1
            continue

        # Headers
        if line.startswith("### "):
            _insert_inline(text_widget, line[4:], "h3")
            text_widget.insert("end", "\n")
        elif line.startswith("## "):
            _insert_inline(text_widget, line[3:], "h2")
            text_widget.insert("end", "\n")
        elif line.startswith("# "):
            _insert_inline(text_widget, line[2:], "h1")
            text_widget.insert("end", "\n")
        # Bullet lists
        elif re.match(r"^(\s*[-*]|\s*\d+\.)\s+", line):
            m = re.match(r"^(\s*[-*]|\s*\d+\.)\s+(.*)", line)
            prefix = "  \u2022 " if m.group(1).strip() in ("-", "*") else f"  {m.group(1).strip()} "
            text_widget.insert("end", prefix)
            _insert_inline(text_widget, m.group(2), "bullet")
            text_widget.insert("end", "\n")
        # Blockquote
        elif line.startswith("> "):
            _insert_inline(text_widget, line[2:], "blockquote")
            text_widget.insert("end", "\n")
        # Blank line
        elif line.strip() == "":
            text_widget.insert("end", "\n")
        # Normal paragraph
        else:
            _insert_inline(text_widget, line, None)
            text_widget.insert("end", "\n")

        i += 1

    text_widget.configure(state="disabled")


def _insert_inline(text_widget: tk.Text, text: str, base_tag: str | None) -> None:
    """Insert text with inline markdown formatting (bold, italic, code, citations)."""
    # Pattern: ***bold italic***, **bold**, *italic*, `code`, [N] citations
    pattern = re.compile(
        r"(\*\*\*(.+?)\*\*\*)"    # bold italic
        r"|(\*\*(.+?)\*\*)"       # bold
        r"|(\*(.+?)\*)"           # italic
        r"|(`(.+?)`)"             # inline code
        r"|(\[(\d+)\])"           # citation
    )
    last = 0
    for m in pattern.finditer(text):
        # Insert text before match
        if m.start() > last:
            plain = text[last:m.start()]
            text_widget.insert("end", plain, base_tag if base_tag else ())

        if m.group(2):      # bold italic
            tags = ("bold_italic",) + ((base_tag,) if base_tag else ())
            text_widget.insert("end", m.group(2), tags)
        elif m.group(4):    # bold
            tags = ("bold",) + ((base_tag,) if base_tag else ())
            text_widget.insert("end", m.group(4), tags)
        elif m.group(6):    # italic
            tags = ("italic",) + ((base_tag,) if base_tag else ())
            text_widget.insert("end", m.group(6), tags)
        elif m.group(8):    # code
            text_widget.insert("end", m.group(8), "code")
        elif m.group(10):   # citation
            text_widget.insert("end", f"[{m.group(10)}]", "citation")

        last = m.end()

    # Trailing text
    if last < len(text):
        remaining = text[last:]
        text_widget.insert("end", remaining, base_tag if base_tag else ())


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class MercuryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mercury — Document Intelligence")
        self.geometry("1200x800")
        self.minsize(900, 600)

        self.doc_store: DocumentStore | None = None
        self.vector_store: VectorStore | None = None
        self._all_records = []  # cached document list
        self._checked: dict[str, bool] = {}  # file_hash -> checked state

        self._build_ui()
        self._set_connected(False)

    # ----- UI construction ------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        self._build_connection_bar()
        self._build_main_area()
        self._build_status_bar()

    def _build_connection_bar(self):
        bar = ttk.Frame(self, padding=6)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(3, weight=0)

        ttk.Label(bar, text="Data Directory:").grid(row=0, column=0, padx=(0, 4))
        self.data_dir_var = tk.StringVar(value=DEFAULT_DATA_DIR)
        ttk.Entry(bar, textvariable=self.data_dir_var, width=40).grid(row=0, column=1, sticky="ew", padx=(0, 2))
        ttk.Button(bar, text="Browse\u2026", width=9, command=self._browse_data_dir).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(bar, text="Collection:").grid(row=0, column=3, padx=(0, 4))
        self.collection_var = tk.StringVar(value=DEFAULT_COLLECTION)
        ttk.Entry(bar, textvariable=self.collection_var, width=18).grid(row=0, column=4, padx=(0, 8))

        self.connect_btn = ttk.Button(bar, text="Connect", command=self._on_connect)
        self.connect_btn.grid(row=0, column=5)

    def _build_main_area(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 2))

        # Left: document list
        left = ttk.Frame(pane, padding=4)
        pane.add(left, weight=1)
        self._build_doc_panel(left)

        # Right: tabbed area
        right = ttk.Frame(pane, padding=4)
        pane.add(right, weight=2)
        self._build_tabs(right)

    def _build_doc_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(parent, text="Documents", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # Filter
        filter_frame = ttk.Frame(parent)
        filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        filter_frame.columnconfigure(1, weight=1)
        ttk.Label(filter_frame, text="Filter:").grid(row=0, column=0, padx=(0, 4))
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(filter_frame, textvariable=self.filter_var).grid(row=0, column=1, sticky="ew")

        # Treeview with checkbox column
        cols = ("name", "pages", "status", "hash")
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.doc_tree = ttk.Treeview(tree_frame, columns=cols, show="tree headings", selectmode="extended")
        self.doc_tree.heading("#0", text="\u2611")
        self.doc_tree.column("#0", width=36, minwidth=36, stretch=False)
        self.doc_tree.heading("name", text="Document Name")
        self.doc_tree.heading("pages", text="Pages")
        self.doc_tree.heading("status", text="Status")
        self.doc_tree.heading("hash", text="File Hash")
        self.doc_tree.column("name", width=180, minwidth=100)
        self.doc_tree.column("pages", width=50, minwidth=40, anchor="center")
        self.doc_tree.column("status", width=70, minwidth=50, anchor="center")
        self.doc_tree.column("hash", width=140, minwidth=80)
        self.doc_tree.grid(row=0, column=0, sticky="nsew")
        self.doc_tree.bind("<Button-1>", self._on_tree_click)
        self.doc_tree.bind("<Double-Button-1>", self._on_tree_double_click)

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.doc_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.doc_tree.configure(yscrollcommand=scroll.set)

        # Buttons
        btn_row = ttk.Frame(parent)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.refresh_btn = ttk.Button(btn_row, text="Refresh", command=self._refresh_docs)
        self.refresh_btn.pack(side="left", padx=(0, 4))
        self.delete_btn = ttk.Button(btn_row, text="Delete Checked", command=self._on_delete)
        self.delete_btn.pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="All", width=4, command=self._check_all).pack(side="left", padx=(0, 2))
        ttk.Button(btn_row, text="None", width=5, command=self._check_none).pack(side="left")

        self.doc_count_label = ttk.Label(parent, text="")
        self.doc_count_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_tabs(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._build_ingest_tab()
        self._build_ask_tab()

    def _build_ingest_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Ingest ")
        tab.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(tab, text="PDF File:").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=4)
        file_frame = ttk.Frame(tab)
        file_frame.grid(row=row, column=1, sticky="ew", pady=4)
        file_frame.columnconfigure(0, weight=1)
        self.ingest_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.ingest_file_var).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(file_frame, text="Browse\u2026", command=self._browse_pdf).grid(row=0, column=1)

        row += 1
        ttk.Label(tab, text="Chunk Size:").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=4)
        self.chunk_size_var = tk.IntVar(value=50)
        chunk_frame = ttk.Frame(tab)
        chunk_frame.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Spinbox(chunk_frame, from_=0, to=500, textvariable=self.chunk_size_var, width=8).pack(side="left")
        ttk.Label(chunk_frame, text="  (pages per batch, 0 = all at once)").pack(side="left")

        row += 1
        self.ingest_btn = ttk.Button(tab, text="Ingest", command=self._on_ingest)
        self.ingest_btn.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 4))

        row += 1
        ttk.Separator(tab, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)

        row += 1
        ttk.Label(tab, text="Ingestion Log", font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w")

        row += 1
        tab.rowconfigure(row, weight=1)
        self.ingest_log = tk.Text(tab, height=12, font=("Consolas", 10), state="disabled",
                                  wrap="word", bg="#fafafa", relief="sunken", borderwidth=1)
        self.ingest_log.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(4, 0))

    def _build_ask_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=" Ask ")
        tab.columnconfigure(0, weight=1)

        # Question
        ttk.Label(tab, text="Question:", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.question_text = tk.Text(tab, height=3, font=("Segoe UI", 11), wrap="word",
                                     relief="sunken", borderwidth=1)
        self.question_text.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.question_text.bind("<Control-Return>", lambda e: self._on_ask())

        # Options row
        opts = ttk.Frame(tab)
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(opts, text="Top K:").pack(side="left", padx=(0, 4))
        self.topk_var = tk.IntVar(value=5)
        ttk.Spinbox(opts, from_=1, to=20, textvariable=self.topk_var, width=4).pack(side="left", padx=(0, 12))

        ttk.Label(opts, text="Window:").pack(side="left", padx=(0, 4))
        self.window_var = tk.IntVar(value=1)
        ttk.Spinbox(opts, from_=0, to=5, textvariable=self.window_var, width=4).pack(side="left", padx=(0, 12))

        self.ask_btn = ttk.Button(opts, text="Ask", command=self._on_ask)
        self.ask_btn.pack(side="left")

        # Answer area
        ttk.Label(tab, text="Answer:", font=("Segoe UI", 10, "bold")).grid(
            row=3, column=0, sticky="w", pady=(0, 2))

        answer_frame = ttk.Frame(tab)
        answer_frame.grid(row=4, column=0, sticky="nsew")
        answer_frame.columnconfigure(0, weight=1)
        answer_frame.rowconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

        self.answer_text = tk.Text(answer_frame, height=12, state="disabled", wrap="word",
                                   relief="sunken", borderwidth=1, bg="#ffffff")
        self.answer_text.grid(row=0, column=0, sticky="nsew")
        answer_scroll = ttk.Scrollbar(answer_frame, orient="vertical", command=self.answer_text.yview)
        answer_scroll.grid(row=0, column=1, sticky="ns")
        self.answer_text.configure(yscrollcommand=answer_scroll.set)

        # Sources
        ttk.Label(tab, text="Sources:", font=("Segoe UI", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(8, 2))
        src_cols = ("doc", "section", "page")
        self.source_tree = ttk.Treeview(tab, columns=src_cols, show="headings", height=4)
        self.source_tree.heading("doc", text="Document")
        self.source_tree.heading("section", text="Section")
        self.source_tree.heading("page", text="Page")
        self.source_tree.column("doc", width=200, minwidth=100)
        self.source_tree.column("section", width=240, minwidth=100)
        self.source_tree.column("page", width=60, minwidth=40, anchor="center")
        self.source_tree.grid(row=6, column=0, sticky="ew", pady=(0, 0))

    def _build_status_bar(self):
        bar = ttk.Frame(self, relief="sunken", padding=(6, 3))
        bar.grid(row=2, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="Ready — not connected")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")

    # ----- State helpers --------------------------------------------------

    def _set_connected(self, connected: bool):
        state = "!disabled" if connected else "disabled"
        for w in (self.refresh_btn, self.delete_btn, self.ingest_btn, self.ask_btn):
            w.state([state])
        if not connected:
            self.doc_tree.delete(*self.doc_tree.get_children())
            self.doc_count_label.configure(text="")

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _log_ingest(self, msg: str):
        self.ingest_log.configure(state="normal")
        self.ingest_log.insert("end", msg + "\n")
        self.ingest_log.see("end")
        self.ingest_log.configure(state="disabled")

    def _run_in_thread(self, target, *args):
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()

    # ----- Connection -----------------------------------------------------

    def _browse_data_dir(self):
        path = filedialog.askdirectory(title="Select Data Directory")
        if path:
            self.data_dir_var.set(path)

    def _on_connect(self):
        data_dir = self.data_dir_var.get().strip()
        collection = self.collection_var.get().strip()
        if not data_dir or not collection:
            messagebox.showwarning("Missing Fields", "Please provide both a data directory and collection name.")
            return

        root = Path(data_dir) / collection
        try:
            self.doc_store = DocumentStore(str(root / "doc_store"))
            self.vector_store = VectorStore(
                collection_name=collection,
                persist_directory=str(root / "chroma"),
            )
            self._set_connected(True)
            self._set_status(f"Connected to {collection}")
            self._refresh_docs()
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect:\n{e}")
            self._set_connected(False)

    # ----- Document list --------------------------------------------------

    def _refresh_docs(self):
        if not self.doc_store:
            return
        try:
            self._all_records = self.doc_store.list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list documents:\n{e}")
            return
        self._apply_filter()

    def _apply_filter(self):
        needle = self.filter_var.get().strip().lower()
        self.doc_tree.delete(*self.doc_tree.get_children())
        filtered = [
            r for r in self._all_records
            if not needle or needle in r.document_name.lower() or needle in r.filename.lower()
        ]
        status_icons = {"complete": "\u2713", "pending": "\u23f3", "failed": "\u2717"}
        for r in filtered:
            checked = self._checked.get(r.file_hash, True)
            cb = "\u2611" if checked else "\u2610"
            self.doc_tree.insert("", "end", iid=r.file_hash, text=cb, values=(
                r.document_name,
                r.page_count,
                status_icons.get(r.status, r.status),
                r.file_hash,
            ))
        checked_count = sum(1 for r in self._all_records if self._checked.get(r.file_hash, True))
        self.doc_count_label.configure(
            text=f"{len(filtered)} document(s) — {checked_count} checked")

    def _on_tree_click(self, event):
        """Toggle checkbox when the #0 (tree) column is clicked."""
        region = self.doc_tree.identify_region(event.x, event.y)
        if region == "tree":
            item = self.doc_tree.identify_row(event.y)
            if item:
                current = self._checked.get(item, True)
                self._checked[item] = not current
                cb = "\u2611" if not current else "\u2610"
                self.doc_tree.item(item, text=cb)
                checked_count = sum(1 for r in self._all_records if self._checked.get(r.file_hash, True))
                self.doc_count_label.configure(
                    text=f"{len(self.doc_tree.get_children())} document(s) — {checked_count} checked")

    def _check_all(self):
        for r in self._all_records:
            self._checked[r.file_hash] = True
        self._apply_filter()

    def _check_none(self):
        for r in self._all_records:
            self._checked[r.file_hash] = False
        self._apply_filter()

    # ----- Rename ---------------------------------------------------------

    def _on_tree_double_click(self, event):
        """Open an inline rename dialog when double-clicking the name column."""
        region = self.doc_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col = self.doc_tree.identify_column(event.x)
        item = self.doc_tree.identify_row(event.y)
        if not item or col != "#1":  # only the name column
            return

        current_name = self.doc_tree.item(item, "values")[0]
        self._show_rename_dialog(item, current_name)

    def _show_rename_dialog(self, file_hash: str, current_name: str):
        dlg = tk.Toplevel(self)
        dlg.title("Rename Document")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="New name:", padding=(12, 10, 12, 2)).pack(anchor="w")
        name_var = tk.StringVar(value=current_name)
        entry = ttk.Entry(dlg, textvariable=name_var, width=45)
        entry.pack(padx=12, pady=(0, 8))
        entry.select_range(0, "end")
        entry.focus_set()

        btn_row = ttk.Frame(dlg)
        btn_row.pack(padx=12, pady=(0, 10), fill="x")

        def _confirm():
            new_name = name_var.get().strip()
            if not new_name:
                messagebox.showwarning("Invalid Name", "Name cannot be empty.", parent=dlg)
                return
            dlg.destroy()
            self._set_status("Renaming\u2026")
            self._run_in_thread(self._rename_worker, file_hash, new_name)

        ttk.Button(btn_row, text="Rename", command=_confirm).pack(side="right", padx=(4, 0))
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="right")
        entry.bind("<Return>", lambda _: _confirm())
        entry.bind("<Escape>", lambda _: dlg.destroy())

        # Centre over parent
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_reqwidth()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _rename_worker(self, file_hash: str, new_name: str):
        try:
            self.doc_store.rename(file_hash, new_name)
            self.vector_store.rename_document(file_hash, new_name)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Rename Error", str(e)))
            self.after(0, lambda: self._set_status("Rename failed"))
            return
        self.after(0, self._refresh_docs)
        self.after(0, lambda: self._set_status(f"Renamed to \"{new_name}\""))

    # ----- Delete ---------------------------------------------------------

    def _on_delete(self):
        checked = [fh for fh, v in self._checked.items() if v]
        if not checked:
            messagebox.showinfo("No Selection", "Please check one or more documents to delete.")
            return
        names = [self.doc_tree.item(fh, "values")[0] for fh in checked if self.doc_tree.exists(fh)]
        msg = "Delete the following document(s)?\n\n" + "\n".join(f"  \u2022 {n}" for n in names)
        if not messagebox.askyesno("Confirm Delete", msg):
            return
        self._set_status("Deleting\u2026")
        self._run_in_thread(self._delete_worker, checked)

    def _delete_worker(self, file_hashes: list[str]):
        for fh in file_hashes:
            try:
                self.vector_store.delete(fh)
                self.doc_store.delete(fh)
                self._checked.pop(fh, None)
            except Exception as e:
                self.after(0, lambda e=e: messagebox.showerror("Delete Error", str(e)))
        self.after(0, self._refresh_docs)
        self.after(0, lambda: self._set_status("Delete complete"))

    # ----- Ingest ---------------------------------------------------------

    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            title="Select PDF to Ingest",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.ingest_file_var.set(path)

    def _on_ingest(self):
        pdf_path = self.ingest_file_var.get().strip()
        if not pdf_path or not Path(pdf_path).exists():
            messagebox.showwarning("Invalid File", "Please select a valid PDF file.")
            return
        if not self.doc_store or not self.vector_store:
            messagebox.showwarning("Not Connected", "Please connect to a collection first.")
            return

        # Clear log
        self.ingest_log.configure(state="normal")
        self.ingest_log.delete("1.0", "end")
        self.ingest_log.configure(state="disabled")

        self.ingest_btn.state(["disabled"])
        self._set_status("Ingesting\u2026")
        chunk_size = self.chunk_size_var.get()
        self._run_in_thread(self._ingest_worker, pdf_path, chunk_size)

    def _ingest_worker(self, pdf_path: str, chunk_size: int):
        try:
            converter = Converter()

            if chunk_size > 0:
                self._ingest_chunked(converter, pdf_path, chunk_size)
            else:
                self._ingest_whole(converter, pdf_path)

            self.after(0, self._refresh_docs)
            self.after(0, lambda: self._set_status("Ingestion complete"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ingestion Error", str(e)))
            self.after(0, lambda: self._set_status("Ingestion failed"))
        finally:
            self.after(0, lambda: self.ingest_btn.state(["!disabled"]))

    def _ingest_whole(self, converter: Converter, pdf_path: str):
        self.after(0, lambda: self._log_ingest("Converting PDF\u2026"))
        dl_doc = converter.convert(pdf_path)

        file_hash = str(dl_doc.origin.binary_hash)
        if self.doc_store.exists(file_hash):
            self.after(0, lambda: self._log_ingest("Document already ingested — skipping."))
            return

        self.after(0, lambda: self._log_ingest(f"Storing document (hash: {file_hash})\u2026"))
        self.doc_store.create(dl_doc, source_pdf_path=pdf_path)

        self.after(0, lambda: self._log_ingest("Chunking\u2026"))
        chunks = Chunker(dl_doc).chunk()

        self.after(0, lambda n=len(chunks): self._log_ingest(f"Embedding {n} chunks\u2026"))
        self.vector_store.create(chunks)
        self.doc_store.set_status(file_hash, "complete")

        self.after(0, lambda n=len(chunks): self._log_ingest(
            f"Done — {dl_doc.name}, {len(dl_doc.pages)} pages, {n} chunks"))

    def _ingest_chunked(self, converter: Converter, pdf_path: str, chunk_size: int):
        total_pages = converter.page_count(pdf_path)
        ranges = [
            (start, min(start + chunk_size - 1, total_pages))
            for start in range(1, total_pages + 1, chunk_size)
        ]
        total_ranges = len(ranges)

        # First range — get document identity
        start, end = ranges[0]
        self.after(0, lambda: self._log_ingest(
            f"Converting pages {start}-{end} (batch 1/{total_ranges})\u2026"))
        first_dl_doc = converter.convert_page_range(pdf_path, start, end)

        file_hash = str(first_dl_doc.origin.binary_hash)
        if self.doc_store.exists(file_hash):
            self.after(0, lambda: self._log_ingest("Document already ingested — skipping."))
            return

        self.after(0, lambda: self._log_ingest(f"Storing document (hash: {file_hash})\u2026"))
        self.doc_store.create(first_dl_doc, source_pdf_path=pdf_path, page_count=total_pages)

        chunk_offset = 0
        total_chunks = 0

        for i, (start, end) in enumerate(ranges):
            dl_doc = first_dl_doc if i == 0 else None
            if dl_doc is None:
                self.after(0, lambda s=start, e=end, idx=i: self._log_ingest(
                    f"Converting pages {s}-{e} (batch {idx+1}/{total_ranges})\u2026"))
                dl_doc = converter.convert_page_range(pdf_path, start, end)

            self.after(0, lambda s=start, e=end: self._log_ingest(
                f"Chunking pages {s}-{e}\u2026"))
            range_chunks = Chunker(dl_doc).chunk()

            # Renumber indices globally
            for chunk in range_chunks:
                new_index = chunk_offset + chunk.index
                chunk.index = new_index
                chunk.id = f"{file_hash}_{new_index}"

            self.after(0, lambda n=len(range_chunks), s=start, e=end: self._log_ingest(
                f"Embedding {n} chunks (pages {s}-{e})\u2026"))
            self.vector_store.create(range_chunks)

            chunk_offset += len(range_chunks)
            total_chunks += len(range_chunks)

        self.doc_store.set_status(file_hash, "complete")
        self.after(0, lambda: self._log_ingest(
            f"Done — {first_dl_doc.name}, {total_pages} pages, {total_chunks} chunks"))

    # ----- Ask ------------------------------------------------------------

    def _on_ask(self):
        question = self.question_text.get("1.0", "end").strip()
        if not question:
            messagebox.showinfo("No Question", "Please type a question.")
            return
        if not self.vector_store:
            messagebox.showwarning("Not Connected", "Please connect to a collection first.")
            return

        # Filter to checked documents (if not all are checked)
        file_hash = None
        checked = [fh for fh, v in self._checked.items() if v]
        all_hashes = [r.file_hash for r in self._all_records]
        if checked and len(checked) < len(all_hashes):
            file_hash = checked[0] if len(checked) == 1 else checked

        self.ask_btn.state(["disabled"])
        self._set_status("Thinking\u2026")

        # Clear answer & sources
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        self.answer_text.configure(state="disabled")
        self.source_tree.delete(*self.source_tree.get_children())

        top_k = self.topk_var.get()
        window = self.window_var.get()
        self._run_in_thread(self._ask_worker, question, top_k, window, file_hash)

    def _ask_worker(self, question: str, top_k: int, window: int, file_hash):
        try:
            self.after(0, lambda: self._set_status("Expanding query\u2026"))
            keyphrases = expand_query(question)

            self.after(0, lambda: self._set_status("Searching\u2026"))
            results = self.vector_store.query(
                query_text=keyphrases,
                top_k=top_k,
                file_hash=file_hash,
                window=window,
            )

            if not results:
                self.after(0, lambda: self._show_answer(
                    "No relevant documents were found for your question.", []))
                return

            self.after(0, lambda: self._set_status("Generating answer\u2026"))
            answer = answer_question(question, results)

            sources = []
            for r in results:
                sources.append({
                    "document": r.chunk.document_name,
                    "section": " > ".join(r.chunk.headings) if r.chunk.headings else "",
                    "page": r.chunk.page_number,
                })

            self.after(0, lambda: self._show_answer(answer, sources))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ask Error", str(e)))
            self.after(0, lambda: self._set_status("Ask failed"))
        finally:
            self.after(0, lambda: self.ask_btn.state(["!disabled"]))

    def _show_answer(self, answer: str, sources: list[dict]):
        render_markdown(self.answer_text, answer)

        self.source_tree.delete(*self.source_tree.get_children())
        seen = set()
        for s in sources:
            key = (s["document"], s.get("section", ""), s.get("page"))
            if key not in seen:
                seen.add(key)
                page_str = str(s["page"]) if s.get("page") is not None else "\u2014"
                self.source_tree.insert("", "end", values=(s["document"], s.get("section", ""), page_str))

        self._set_status("Ready")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = MercuryGUI()
    app.mainloop()
