# Git LFS Setup Guide

## What is Git LFS?

Git LFS (Large File Storage) stores large files outside the main repository, keeping your repo fast and small.

## Setup Instructions

### 1. Install Git LFS

**Windows:**
```bash
# Download from: https://git-lfs.github.com/
# Or use package manager:
winget install Git.Git-LFS
```

**Linux:**
```bash
sudo apt-get install git-lfs  # Debian/Ubuntu
# or
sudo yum install git-lfs      # CentOS/RHEL
```

**Mac:**
```bash
brew install git-lfs
```

### 2. Initialize Git LFS

```bash
# One-time setup (per machine)
git lfs install
```

### 3. Verify Setup

The `.gitattributes` file is already configured to track:
- All files in `examples/showcase/` directory
- VTI files (`*.vti`)
- Binary structure function files (`*.bin`)

### 4. Use Git Normally

```bash
# Add files normally
git add examples/showcase/
git commit -m "Add example data"
git push
```

Git LFS will automatically handle large files!

## Verify LFS is Working

```bash
# Check which files are tracked by LFS
git lfs ls-files

# Check LFS status
git lfs status
```

## GitHub Limits

- **Free accounts**: 10 GB storage, 10 GB/month bandwidth
- **Team/Enterprise**: 250 GB storage, 250 GB/month bandwidth

For example data (~30-120 MB), this is well within free limits!

## Troubleshooting

### If files aren't being tracked:

1. Check `.gitattributes` exists and has correct paths
2. Make sure Git LFS is installed: `git lfs version`
3. Re-initialize: `git lfs install`
4. Re-add files: `git add examples/showcase/`

### If you get "Git LFS not found":

```bash
# Install Git LFS first, then:
git lfs install
```

## More Information

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub Git LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

