```python
markdown_content = """# Managing Zotero PDFs with Git LFS: A Complete Tutorial

This guide outlines the hybrid architecture for managing a Zotero library. By using Zotero's native servers to sync lightweight metadata (like tags, annotations, and citations) and utilizing Git Large File Storage (LFS) to version-control the heavy PDF files, you create a robust, easily portable, and Obsidian-friendly research environment.

---

## 🛠 Prerequisites

1. **Zotero 7** installed and logged into your account.
2. **Git** and **Git LFS** installed on your system.
3. The **ZotMoov** plugin (`.xpi` file) downloaded. 
   *(Note: If downloading via Firefox, hold `Alt` and click the link, or right-click and "Save Link As..." to prevent Firefox from trying to install it as a browser extension).*

---

## Step 1: Initialize the Git LFS Repository

You must separate your Zotero database from your PDFs. **Never put your `zotero.sqlite` database in a Git repository.** Based on our setup, we will create a dedicated folder for the PDFs at `/home/utsab/zotero-library`.

Open your terminal and run:


```

```text
File generated successfully: zotero_git_lfs_tutorial.md

```bash
# Navigate to or create your library folder
cd /home/utsab/zotero-library
git init

# Initialize Git LFS
git lfs install

# Tell Git LFS to track all PDF files
git lfs track "*.pdf"

# Commit the tracking rules
git add .gitattributes
git commit -m "Initialize Git LFS for PDFs"

```

---

## Step 2: Configure Zotero Native Settings

We need to tell Zotero to handle metadata but stop syncing files, and configure how it renames and locates linked PDFs.

1. **Disable File Syncing:**
* Go to **Edit > Preferences** (or Zotero > Settings) **> Sync**.
* Keep "Sync automatically" and "Sync full-text content" **CHECKED**.
* **UNCHECK** "Sync attachment files in My Library".


2. **Set the Base Directory (Crucial for Portability):**
* Go to **Settings > Advanced > Files and Folders**.
* Under **Linked Attachment Base Directory**, click "Choose..." and select `/home/utsab/zotero-library`.
* *Ensure your Data Directory Location (e.g., `/home/utsab/Zotero`) remains completely separate from your Base Directory!*


3. **Enable Native Renaming:**
* Go to **Settings > General**.
* Scroll down to the **File Renaming** section.
* Check **"Automatically rename locally added files"**.
* Enter your preferred format (e.g., `{{author}}_{{year}}_{{title}}`).



---

## Step 3: Install and Configure ZotMoov

ZotMoov bridges the gap by automatically moving newly downloaded papers out of Zotero's hidden storage and into your Git folder.

1. **Install the Plugin:**
* Open **Tools > Add-ons**.
* Drag and drop the downloaded `zotmoov.xpi` file directly into the middle of the Add-ons window to bypass any OS confusion.
* Restart Zotero.


2. **Configure ZotMoov:**
* Open ZotMoov's settings via **Tools > ZotMoov Preferences**.
* Set the **Directory to Move/Copy Files To** to `/home/utsab/zotero-library`.
* Set **File Behavior** to `Move`.
* Check **"Automatically Move/Copy Files When Added"**.
* **CRITICAL:** **UNCHECK** `"Automatically Move/Copy Files to Subdirectory"`. (Leaving this checked will create a messy folder-per-file structure instead of relying on Zotero's clean native renaming).


3. **Migrate Your Existing Library:**
* Go to your main Zotero library pane.
* Click one paper, then press `Ctrl + A` to select all.
* Right-click the highlighted list and select **ZotMoov: Move Selected to Directory**.
* ZotMoov will rename and move all existing PDFs into your Git folder.



---

## Step 4: Push to GitHub

With your PDFs neatly organized in `/home/utsab/zotero-library`, you can push them to GitHub.

*Warning: Ensure your GitHub repository is set to **Private** to avoid copyright issues with academic papers. Also note that GitHub's free tier has a **1 GB LFS limit**.*

```bash
# Link your local folder to your new private GitHub repository
git remote add origin [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)

# Set branch name to main
git branch -M main

# Add, commit, and push your PDFs
git add .
git commit -m "Initial Zotero library backup"
git push -u origin main

```

---

## Step 5: Restoring on a New Device

To replicate this environment perfectly on a new machine:

1. Install Zotero and log in to the **Sync** tab to pull down your metadata database (tags, collections, annotations).
2. Clone your Git repository to the exact same path (or a new path of your choosing):
```bash
git clone <your-repo-url> /home/utsab/zotero-library

```


3. In Zotero, go to **Settings > Advanced > Files and Folders** and set the **Linked Attachment Base Directory** to your cloned folder. Zotero will instantly remap the database to the local files via relative paths.

---

---

## ❓ Frequently Asked Questions (Q&A)

### 1. Are my PDF highlights and annotations saved on GitHub?

**It depends on your reader.** - If you use **Zotero's built-in PDF reader**, annotations are saved as raw text inside the `zotero.sqlite` database. They are synced via Zotero's unlimited server sync, *not* Git.

* If you open the linked file in an **external PDF reader** (like Evince or Acrobat), your highlights are embedded directly into the binary PDF. Git LFS will detect the file change and sync it to GitHub.

### 2. Does "Sync full-text content" eat into my 1GB Git LFS limit?

**No.** "Sync full-text content" is a Zotero setting that extracts the raw text from your PDFs to build a search index. This text index is synced to Zotero's servers (which offer unlimited data syncing for metadata), completely bypassing GitHub and Git LFS.

### 3. Why did we uncheck the ZotMoov "Subdirectory" option?

If checked, ZotMoov intercepts the formatting rules (like `{%a_}{%y_}{%t}`) and creates a *brand-new folder* with that name, placing an poorly-named PDF inside it. By unchecking it, ZotMoov dumps the PDF directly into the root directory, allowing Zotero's native engine to handle the actual file renaming seamlessly.

### 4. Why couldn't I install the `.xpi` file easily?

Zotero shares architecture with Mozilla Firefox. When you double-click an `.xpi` file in your OS, or try to download it natively in Firefox, your computer assumes it is a web browser extension. It tries to install it to Firefox, which fails because the code is for Zotero. Dragging and dropping into Zotero's Add-on manager bypasses this OS-level confusion entirely.
"""

with open("zotero_git_lfs_tutorial.md", "w") as f:
f.write(markdown_content)

print("File generated successfully: zotero_git_lfs_tutorial.md")

```
Your MD file is ready
[file-tag: code-generated-file-0-1779166227303895127]

The Markdown file contains the complete tutorial spanning everything from initializing Git LFS to configuring Zotero's settings and fixing the ZotMoov plugin behavior. It also includes the specific directory paths you are using on your machine and addresses the exact edge cases we discussed in the Q&A section at the end.

```