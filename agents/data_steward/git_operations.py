"""
Git operations for ActionExecutor
"""
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import re

from .path_resolver import format_error
from agents.shared.config import GIT_OPERATION_TIMEOUT, GIT_PUSH_PULL_TIMEOUT, GIT_LOG_DEFAULT_LIMIT


def execute_git_operation(action: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Execute Git operations with confirmation for write operations"""
    operation = action.get("operation", "").lower()
    confirmed = action.get("confirmed", False)
    
    if not operation:
        return {"success": False, "message": "Git operation required"}
    
    try:
        # Expanded safe operations list
        safe_operations = [
            "status", "add", "commit", "push", "pull", "log", "diff", "branch",
            "branch_create", "branch_switch", "branch_delete", "branch_merge",
            "stash", "stash_pop", "stash_apply", "stash_list", "stash_drop",
            "cherry_pick", "rebase", "tag", "tag_create", "tag_list", "tag_push", "tag_delete",
            "remote_list", "remote_add", "remote_remove",
            "conflict_detect", "restore", "reset_soft", "suggest"
        ]
        if operation not in safe_operations:
            return {"success": False, "message": f"Git operation '{operation}' not supported"}
        
        # Safety: Block dangerous git commands
        dangerous_ops = ["reset --hard", "clean -fd", "push --force", "push -f", "force-with-lease"]
        if any(danger in str(action).lower() for danger in dangerous_ops):
            return {"success": False, "message": "⚠️ Dangerous git operation blocked for safety"}
        
        # Read-only operations (no confirmation needed)
        read_only_ops = ["status", "log", "diff", "branch", "stash_list", "tag_list", "remote_list", "conflict_detect", "suggest"]
        # Write operations (require confirmation)
        write_ops = ["add", "commit", "push", "branch_create", "branch_delete", "branch_merge",
                     "stash", "stash_pop", "stash_apply", "stash_drop",
                     "cherry_pick", "rebase", "tag_create", "tag_push", "tag_delete",
                     "remote_add", "remote_remove", "restore", "reset_soft"]
        # Destructive operations (also require confirmation)
        destructive_ops = ["pull", "branch_switch"]
        
        # Check if operation requires confirmation
        requires_confirmation = (operation in write_ops or operation in destructive_ops) and operation not in read_only_ops
        
        if requires_confirmation and not confirmed:
            # Return confirmation request with operation-specific details
            op_display_map = {
                "add": "stage files", "commit": "commit changes", "push": "push to remote", "pull": "pull from remote",
                "branch_create": "create branch", "branch_switch": "switch branch", "branch_delete": "delete branch",
                "branch_merge": "merge branch", "stash": "stash changes", "stash_pop": "pop stash",
                "cherry_pick": "cherry-pick commit", "rebase": "rebase", "tag_create": "create tag",
                "tag_push": "push tag", "tag_delete": "delete tag", "remote_add": "add remote",
                "remote_remove": "remove remote", "restore": "restore file", "reset_soft": "undo commit"
            }
            op_display = op_display_map.get(operation, operation)
            
            details = {}
            for key in ["files", "message", "branch", "commit_hash", "tag_name", "remote_name", "remote_url", "target_branch"]:
                if key in action:
                    details[key] = action[key]
            
            return {
                "success": False,
                "message": f"Confirmation required to {op_display}",
                "requires_confirmation": True,
                "action": "git_operation",
                "data": {"operation": operation, "details": details}
            }
        
        result_output = ""
        
        # Basic operations
        if operation == "status":
            result = subprocess.run(
                ["git", "status", "--short", "--branch"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else result.stderr
        
        elif operation == "add":
            files = action.get("files", [])
            if not files:
                return {"success": False, "message": "Files required for 'add' operation"}
            result = subprocess.run(
                ["git", "add"] + files,
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Staged {len(files)} file(s)" if result.returncode == 0 else result.stderr
        
        elif operation == "commit":
            message = action.get("message", "Auto-commit from AI assistant")
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else result.stderr
        
        elif operation == "push":
            branch = action.get("branch", "")
            remote = action.get("remote", "origin")
            cmd = ["git", "push", remote]
            if branch:
                cmd.append(branch)
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_root), timeout=GIT_PUSH_PULL_TIMEOUT
            )
            result_output = result.stdout if result.stdout else result.stderr
        
        elif operation == "pull":
            result = subprocess.run(
                ["git", "pull"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_PUSH_PULL_TIMEOUT
            )
            result_output = result.stdout if result.stdout else result.stderr
        
        # Advanced log with filters
        elif operation == "log":
            cmd = ["git", "log", "--oneline"]
            limit = action.get("limit", GIT_LOG_DEFAULT_LIMIT)
            cmd.append(f"-{limit}")
            if "author" in action:
                cmd.extend(["--author", action["author"]])
            if "since" in action:
                cmd.extend(["--since", action["since"]])
            if "until" in action:
                cmd.extend(["--until", action["until"]])
            if "grep" in action:
                cmd.extend(["--grep", action["grep"]])
            if "file" in action:
                cmd.extend(["--", action["file"]])
            if action.get("graph", False):
                cmd.insert(2, "--graph")
                cmd.insert(2, "--all")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else result.stderr
        
        # Advanced diff
        elif operation == "diff":
            cmd = ["git", "diff"]
            if action.get("staged", False):
                cmd.append("--staged")
            if "files" in action:
                cmd.extend(["--"] + action["files"])
            if "from_ref" in action and "to_ref" in action:
                cmd = ["git", "diff", action["from_ref"], action["to_ref"]]
            elif "ref" in action:
                cmd = ["git", "diff", action["ref"]]
            if action.get("stat", False):
                cmd.append("--stat")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else "No changes"
        
        # Branch operations
        elif operation == "branch":
            result = subprocess.run(
                ["git", "branch", "-a"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else result.stderr
        
        elif operation == "branch_create":
            branch_name = action.get("branch", action.get("name"))
            if not branch_name:
                return {"success": False, "message": "Branch name required"}
            result = subprocess.run(
                ["git", "branch", branch_name],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            if result.returncode != 0:
                return {"success": False, "message": f"Failed: {result.stderr}"}
            if action.get("switch", False):
                subprocess.run(["git", "checkout", branch_name], cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT)
                result_output = f"✅ Created and switched to '{branch_name}'"
            else:
                result_output = f"✅ Created branch '{branch_name}'"
        
        elif operation == "branch_switch":
            branch_name = action.get("branch", action.get("name"))
            if not branch_name:
                return {"success": False, "message": "Branch name required"}
            result = subprocess.run(
                ["git", "checkout", branch_name],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Switched to '{branch_name}'" if result.returncode == 0 else result.stderr
        
        elif operation == "branch_delete":
            branch_name = action.get("branch", action.get("name"))
            if not branch_name:
                return {"success": False, "message": "Branch name required"}
            flag = "-D" if action.get("force", False) else "-d"
            result = subprocess.run(
                ["git", "branch", flag, branch_name],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Deleted branch '{branch_name}'" if result.returncode == 0 else result.stderr
        
        elif operation == "branch_merge":
            branch_name = action.get("branch", action.get("name"))
            if not branch_name:
                return {"success": False, "message": "Branch name required"}
            result = subprocess.run(
                ["git", "merge", branch_name],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            if "CONFLICT" in result.stdout or "CONFLICT" in result.stderr:
                return {"success": False, "message": f"⚠️ Merge conflict detected. Resolve manually."}
            result_output = result.stdout if result.returncode == 0 else result.stderr
        
        # Stash operations
        elif operation == "stash":
            message = action.get("message", "")
            cmd = ["git", "stash", "push"]
            if message:
                cmd.extend(["-m", message])
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if "No local changes" not in result.stdout else "No changes to stash"
        
        elif operation == "stash_list":
            result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else "No stashes"
        
        elif operation == "stash_pop":
            index = action.get("index", 0)
            result = subprocess.run(
                ["git", "stash", "pop", f"stash@{{{index}}}"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.returncode == 0 else result.stderr
        
        elif operation == "stash_apply":
            index = action.get("index", 0)
            result = subprocess.run(
                ["git", "stash", "apply", f"stash@{{{index}}}"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.returncode == 0 else result.stderr
        
        elif operation == "stash_drop":
            index = action.get("index", 0)
            result = subprocess.run(
                ["git", "stash", "drop", f"stash@{{{index}}}"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Stash deleted" if result.returncode == 0 else result.stderr
        
        # Tag operations
        elif operation == "tag" or operation == "tag_list":
            result = subprocess.run(
                ["git", "tag", "-l"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else "No tags"
        
        elif operation == "tag_create":
            tag_name = action.get("tag", action.get("name"))
            if not tag_name:
                return {"success": False, "message": "Tag name required"}
            message = action.get("message", "")
            cmd = ["git", "tag"]
            if message:
                cmd.extend(["-a", tag_name, "-m", message])
            else:
                cmd.append(tag_name)
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Created tag '{tag_name}'" if result.returncode == 0 else result.stderr
        
        elif operation == "tag_push":
            tag_name = action.get("tag", action.get("name"))
            remote = action.get("remote", "origin")
            cmd = ["git", "push", remote]
            if tag_name:
                cmd.append(tag_name)
            else:
                cmd.append("--tags")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(project_root), timeout=GIT_PUSH_PULL_TIMEOUT
            )
            result_output = result.stdout if result.returncode == 0 else result.stderr
        
        elif operation == "tag_delete":
            tag_name = action.get("tag", action.get("name"))
            if not tag_name:
                return {"success": False, "message": "Tag name required"}
            result = subprocess.run(
                ["git", "tag", "-d", tag_name],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Deleted tag '{tag_name}'" if result.returncode == 0 else result.stderr
        
        # Remote operations
        elif operation == "remote_list":
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = result.stdout if result.stdout else "No remotes"
        
        elif operation == "remote_add":
            name = action.get("name", action.get("remote_name"))
            url = action.get("url", action.get("remote_url"))
            if not name or not url:
                return {"success": False, "message": "Remote name and URL required"}
            result = subprocess.run(
                ["git", "remote", "add", name, url],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Added remote '{name}'" if result.returncode == 0 else result.stderr
        
        elif operation == "remote_remove":
            name = action.get("name", action.get("remote_name"))
            if not name:
                return {"success": False, "message": "Remote name required"}
            result = subprocess.run(
                ["git", "remote", "remove", name],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Removed remote '{name}'" if result.returncode == 0 else result.stderr
        
        # Advanced operations
        elif operation == "cherry_pick":
            commit_hash = action.get("commit", action.get("commit_hash"))
            if not commit_hash:
                return {"success": False, "message": "Commit hash required"}
            result = subprocess.run(
                ["git", "cherry-pick", commit_hash],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            if "CONFLICT" in result.stdout or "CONFLICT" in result.stderr:
                return {"success": False, "message": f"⚠️ Cherry-pick conflict. Resolve manually."}
            result_output = result.stdout if result.returncode == 0 else result.stderr
        
        elif operation == "rebase":
            target_branch = action.get("branch", action.get("target_branch", "main"))
            result = subprocess.run(
                ["git", "rebase", target_branch],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            if "CONFLICT" in result.stdout or "CONFLICT" in result.stderr:
                return {"success": False, "message": f"⚠️ Rebase conflict. Resolve then: git rebase --continue"}
            result_output = result.stdout if result.returncode == 0 else result.stderr
        
        elif operation == "conflict_detect":
            status_result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            conflicted_files = [line[3:].strip() for line in status_result.stdout.split('\n') 
                              if line.startswith('UU ') or line.startswith('AA ')]
            if not conflicted_files:
                return {"success": True, "message": "✅ No conflicts", "data": {"conflicts": False, "files": []}}
            return {"success": True, "message": f"⚠️ {len(conflicted_files)} file(s) with conflicts", 
                   "data": {"conflicts": True, "files": conflicted_files}}
        
        elif operation == "restore":
            files = action.get("files", [])
            if not files:
                return {"success": False, "message": "Files required"}
            result = subprocess.run(
                ["git", "restore"] + files,
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Restored {len(files)} file(s)" if result.returncode == 0 else result.stderr
        
        elif operation == "reset_soft":
            commits = action.get("commits", 1)
            result = subprocess.run(
                ["git", "reset", "--soft", f"HEAD~{commits}"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            result_output = f"✅ Undid last {commits} commit(s)" if result.returncode == 0 else result.stderr
        
        elif operation == "suggest":
            # Smart commit message suggestion
            status_result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, cwd=str(project_root), timeout=GIT_OPERATION_TIMEOUT
            )
            status = status_result.stdout
            suggestions = []
            if re.search(r'^\s*[MADRCU?]', status, re.MULTILINE):
                suggestions.append("💡 Uncommitted changes detected")
            if '??' in status:
                suggestions.append("💡 Untracked files present")
            if 'ahead' in status.lower():
                suggestions.append("💡 Branch ahead of remote - consider push")
            result_output = "\n".join(suggestions) if suggestions else "✅ Working tree clean"
        
        else:
            return {"success": False, "message": f"Operation '{operation}' not implemented"}
        
        return {
            "success": True,
            "message": f"Git {operation} completed",
            "data": {"output": result_output}
        }
    
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Git operation timed out"}
    except Exception as e:
        return format_error("Git operation error", e)
