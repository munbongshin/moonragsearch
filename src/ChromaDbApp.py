import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import chromadb
from chromadb.config import Settings
import os, io,time, re, tempfile
import uuid
import logging
import PyPDF2, docx 
from openpyxl import load_workbook
from pptx import Presentation
import threading
import json

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
from sentence_transformers import SentenceTransformer

from typing import Union, Optional, List, IO

import win32com.client as win32
import win32gui
import win32con
import pythoncom
from TextSummarizer import TextSummarizer
from bs4 import BeautifulSoup
import markdown

from ExtractTextFromFile import ExtractTextFromFile as ETF
from ChromaDbManager  import ChromaDbManager 
   

class ChromaDbApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChromaDB Manager")
        self.geometry("800x600")
        self.db_manager = ChromaDbManager()
        self.db_location = self.db_manager.persist_directory
        self.create_widgets()
        
        
    def create_widgets(self):
        # DB Location Frame
        location_frame = ttk.LabelFrame(self, text="ChromaDB Location")
        location_frame.pack(padx=10, pady=10, fill=tk.X)

        self.location_var = tk.StringVar(value=self.db_location)
        self.location_entry = ttk.Entry(location_frame, textvariable=self.location_var, width=50, state='readonly')
        self.location_entry.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)

        ttk.Button(location_frame, text="Browse", command=self.browse_location).pack(side=tk.LEFT, padx=5)
        ttk.Button(location_frame, text="Set Location", command=self.set_db_location).pack(side=tk.LEFT, padx=5)

        # Operation Frame
        operation_frame = ttk.LabelFrame(self, text="Operations")
        operation_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.operation_combo = ttk.Combobox(operation_frame, values=["Create Collection", "Delete Collection", "List Collections", "View Collection Content", "Search Collection", "Store Text"])
        self.operation_combo.pack(padx=5, pady=5)
        self.operation_combo.bind("<<ComboboxSelected>>", self.on_operation_selected)

        self.operation_frame = ttk.Frame(operation_frame)
        self.operation_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
               
        
        # 프로그레스 바 추가
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=10, fill=tk.X, padx=10)
        self.progress_bar.pack_forget()  # 초기에는 숨김

    def browse_location(self):
        new_location = filedialog.askdirectory(initialdir=self.db_location)
        if new_location:
            self.db_location = new_location
            self.location_var.set(self.db_location)

    def set_db_location(self):
        if os.path.exists(self.db_location):
            try:
                self.db_manager.set_persist_directory(self.db_location)
                messagebox.showinfo("Success", f"ChromaDB location set to: {self.db_location}")
            except Exception as e:
                messagebox.showerror("Error", f"Error setting ChromaDB location: {str(e)}")
                self.db_location = self.db_manager.persist_directory
                self.location_var.set(self.db_location)
        else:
            messagebox.showerror("Error", f"The specified path does not exist: {self.db_location}")
            self.db_location = self.db_manager.persist_directory
            self.location_var.set(self.db_location)

    def on_operation_selected(self, event):
        operation = self.operation_combo.get()
        for widget in self.operation_frame.winfo_children():
            widget.destroy()

        if operation == "Create Collection":
            self.setup_create_collection()
        elif operation == "Delete Collection":
            self.setup_delete_collection()
        elif operation == "List Collections":
            self.list_collections()
        elif operation == "View Collection Content":
            self.setup_view_collection()
        elif operation == "Store Text":
            self.setup_store_text()
        elif operation == "Search Collection":
            self.setup_search_collection()

    def setup_store_text(self):
        ttk.Label(self.operation_frame, text="File:").pack(pady=5)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(self.operation_frame, textvariable=self.file_path_var, state='readonly', width=50)
        self.file_path_entry.pack(pady=5)
        ttk.Button(self.operation_frame, text="Browse", command=self.browse_file).pack(pady=5)

        ttk.Label(self.operation_frame, text="Extracted Text:").pack(pady=5)
        self.text_entry = tk.Text(self.operation_frame, height=10)
        self.text_entry.pack(pady=5)

        self.store_button = ttk.Button(self.operation_frame, text="Store", command=self.store_text)
        self.store_button.pack(pady=5)        


    def select_collection(self):
        collections = self.db_manager.get_list_collections()
        collections.append("Create New Collection")
        
        selection = simpledialog.askstring(
            "Select Collection",
            "Choose a collection or create a new one:",
            initialvalue=collections[0] if collections else None
        )

        if selection == "Create New Collection":
            new_collection = simpledialog.askstring("New Collection", "Enter name for new collection:")
            if new_collection:
                return new_collection
            else:
                return None
        
        return selection

    def setup_create_collection(self):
        ttk.Label(self.operation_frame, text="Collection Name:").pack(pady=5)
        self.create_entry = ttk.Entry(self.operation_frame)
        self.create_entry.pack(pady=5)
        ttk.Button(self.operation_frame, text="Create", command=self.create_collection).pack(pady=5)

    def create_collection(self):
        name = self.create_entry.get()
        result = self.db_manager.create_collection(name)
        messagebox.showinfo("Result", result)

    def setup_delete_collection(self):
        collections = self.db_manager.get_list_collections()
        ttk.Label(self.operation_frame, text="Select Collection:").pack(pady=5)
        self.delete_combo = ttk.Combobox(self.operation_frame, values=collections)
        self.delete_combo.pack(pady=5)
        ttk.Button(self.operation_frame, text="Delete", command=self.delete_collection).pack(pady=5)

    def delete_collection(self):
        name = self.delete_combo.get()
        if name:
            result = self.db_manager.delete_collection(name)
            messagebox.showinfo("Result", result)
        else:
            messagebox.showwarning("Warning", "Please select a collection to delete.")

    def list_collections(self):
        collections = self.db_manager.get_list_collections()
        text = tk.Text(self.operation_frame, height=10, width=50)
        text.pack(pady=5)
        for collection in collections:
            text.insert(tk.END, f"- {collection}\n")
        text.config(state=tk.DISABLED)

    def setup_view_collection(self):
        collections = self.db_manager.get_list_collections()
        ttk.Label(self.operation_frame, text="Select Collection:").pack(pady=5)
        self.view_combo = ttk.Combobox(self.operation_frame, values=collections)
        self.view_combo.pack(pady=5)
        ttk.Button(self.operation_frame, text="View", command=self.view_collection).pack(pady=5)
        
    def view_collection(self):
        name = self.view_combo.get()
        print(f"view collectio name : {name}")
        if name:
            content = self.db_manager.view_collection_content(name)
            view_window = tk.Toplevel(self)
            view_window.title(f"Content of {name}")
            text = tk.Text(view_window, wrap=tk.WORD)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            if isinstance(content, list):
                for item in content:
                    text.insert(tk.END, f"{item}\n\n")
            else:
                text.insert(tk.END, content)
            text.config(state=tk.DISABLED)
        else:
            messagebox.showwarning("Warning", "Please select a collection to view.")
            
    def setup_store_text(self):
        collections = self.db_manager.get_list_collections()
        ttk.Label(self.operation_frame, text="Collection Name:").pack(pady=5)
        self.store_collection_entry = ttk.Combobox(self.operation_frame, values=collections)
        self.store_collection_entry.pack(pady=5)

        ttk.Label(self.operation_frame, text="File:").pack(pady=5)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(self.operation_frame, textvariable=self.file_path_var, state='readonly', width=50)
        self.file_path_entry.pack(pady=5)
        ttk.Button(self.operation_frame, text="Browse", command=self.browse_file).pack(pady=5)
        ttk.Label(self.operation_frame, text="Extracted Text:").pack(pady=5)
        self.text_entry = tk.Text(self.operation_frame, height=10)
        self.text_entry.pack(pady=5)
        
        ttk.Button(self.operation_frame, text="Store", command=self.store_text).pack(pady=5)
        

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Supported files", "*.pdf;*.docx;*.xlsx;*.pptx;*.ppt;*.hwp,*hwpx"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Excel workbooks", "*.xlsx"),
                ("PowerPoint presentations", "*.pptx"),
                ("Hwp files", "*.hwp")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
            try:
                extracted_text = self.extract_text_from_file(file_path,file_name=file_path)
                self.text_entry.delete("1.0", tk.END)
                
                # 추출된 텍스트를 적절히 표시
                if isinstance(extracted_text, list):
                    for doc in extracted_text:
                        page_number = doc.metadata.get('page', '')  # 또는 'page_number'
                        content = doc.page_content
                        self.text_entry.insert(tk.END, f"Page {page_number}:\n{content}\n\n")
                elif isinstance(extracted_text, Document):
                    page_number = extracted_text.metadata.get('page', '')  # 또는 'page_number'
                    content = extracted_text.page_content
                    self.text_entry.insert(tk.END, f"Page {page_number}:\n{content}\n")
                else:
                    self.text_entry.insert(tk.END, str(extracted_text))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to extract text: {str(e)}")

    def store_text(self):
        collection_name = self.store_collection_entry.get()
        file_path = self.file_path_var.get()
        #text = self.text_entry.get("1.0", tk.END).strip()
        text = self.extract_text_from_file(file_path, file_name=file_path)
       
        if not collection_name or not file_path or not text:
            messagebox.showwarning("Warning", "Please fill all fields.")
            return
        #self.progress_bar.pack(pady=10, fill=tk.X, padx=10)
        #self.progress_var.set(0)

        # 여기서 text를 직접 전달합니다.
        try:
            if not text:
                raise ValueError("Text is empty")

            # 텍스트 전처리 (필요한 경우)
            processed_text = text  # 여기서 필요한 전처리를 수행할 수 있습니다.

            # 처리된 텍스트를 저장 및 임베딩
            #print("처리된 텍스트를 저장 및 임베딩 시작")
            chunks_stored = self.db_manager.split_embed_docs_store(processed_text, os.path.basename(file_path), collection_name)
            
            messagebox.showinfo("Success", f"Stored and embedded {chunks_stored} chunks in collection '{collection_name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to store and embed text: {str(e)}")
            
    def setup_search_collection(self):
        collections = self.db_manager.get_list_collections()
        ttk.Label(self.operation_frame, text="Select Collection:").pack(pady=5)
        self.search_collection_combo = ttk.Combobox(self.operation_frame, values=collections)
        self.search_collection_combo.pack(pady=5)

        ttk.Label(self.operation_frame, text="Search Query:").pack(pady=5)
        self.search_query_entry = ttk.Entry(self.operation_frame, width=50)
        self.search_query_entry.pack(pady=5)

        ttk.Button(self.operation_frame, text="Search", command=self.perform_search).pack(pady=5)

        self.search_results_text = tk.Text(self.operation_frame, height=15, width=70)
        self.search_results_text.pack(pady=5)

    def perform_search(self):
        collection_name = self.search_collection_combo.get()
        query = self.search_query_entry.get()

        if not collection_name or not query:
            messagebox.showwarning("Warning", "Please select a collection and enter a search query.")
            return

        try:
            results = self.db_manager.search_collection(collection_name, query,self.db_manager.docnum)
            self.display_search_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to perform search: {str(e)}")

    def display_search_results(self, results):
        self.search_results_text.delete('1.0', tk.END)
        if not results:
            self.search_results_text.insert(tk.END, "No results found.")
        else:
            for i, result in enumerate(results, 1):
                self.search_results_text.insert(tk.END, f"Result {i}:\n")
                self.search_results_text.insert(tk.END, f"Content: {result['page_content']}\n")
                self.search_results_text.insert(tk.END, f"Metadata: {result['metadata']}\n")
                self.search_results_text.insert(tk.END, f"Score: {result['score']}\n\n")
  
if __name__ == "__main__":
    app = ChromaDbApp()
    app.mainloop()



 