# Stacked PRs Workflow

## The Problem with "Create All, Review Later"

Creating a full stack of PRs and then addressing all review comments at once leads to:
- Cascading rebases through the entire stack
- Merge conflicts when squash-merging into main
- Cherry-pick recovery needed for later PRs
- Wasted effort if early PRs need significant changes

## Better Approach: Wait for Review Before Continuing

AI/automated reviews typically complete within minutes. Use this workflow:

### Step-by-Step Process

```
1. Create PR #1 (base: main)
2. WAIT for review comments (2-5 minutes)
3. Address review comments on PR #1
4. Push fixes, wait for CI
5. Once PR #1 is approved/green:
   - Create PR #2 (base: feat/layer-1)
   - WAIT for review comments
   - Address and push fixes
6. Repeat for each layer
```

### Why This Works Better

| Approach | Rebases Needed | Conflict Risk | Recovery Effort |
|----------|---------------|---------------|-----------------|
| Wait between PRs | 0 | Low | None |
| Create all, review later | N×(N-1)/2 | High | Cherry-picks |

### Practical Example

```bash
# Create and push PR #1
git checkout -b feat/ci-setup
# ... implement ...
git push -u origin feat/ci-setup
gh pr create --base main --title "feat(ci): add CI workflow"

# WAIT HERE - don't create PR #2 yet
# Check for review comments in 2-5 minutes
gh pr view 1 --comments

# Address any review comments
# ... make fixes ...
git commit -m "fix: address review comments"
git push

# Only NOW create PR #2
git checkout -b feat/artifacts
# ... implement ...
git push -u origin feat/artifacts
gh pr create --base feat/ci-setup --title "feat(artifacts): add schema"

# WAIT for review on PR #2 before creating PR #3
```

## When You Must Create Multiple PRs Upfront

If you need to create the full stack before reviews (e.g., demonstrating architecture):

1. **Create minimal PRs**: Keep each PR as small as possible
2. **Document dependencies**: Note in PR descriptions which PRs must merge first
3. **Address reviews bottom-up**: Always fix PR #1 before touching PR #2
4. **Rebase immediately**: After fixing PR N, rebase all downstream PRs
5. **Merge one at a time**:
   - Merge PR #1
   - Rebase PR #2 onto main, push, merge
   - Repeat

### Recovery from Conflicts

If later PRs have conflicts after earlier ones merged:

```bash
# Find commits unique to this PR (not from earlier PRs)
git log --oneline <previous-pr-branch>..HEAD

# Create clean branch from main
git checkout origin/main -b feat/layer-clean

# Cherry-pick only this PR's commits
git cherry-pick <commit1> <commit2>

# Force-push to fix the PR
git push origin feat/layer-clean:feat/layer --force-with-lease
```

## Summary

**Default workflow**: Create PR → Wait for review → Fix → Create next PR

**Stacked PRs are a last resort**, not a default pattern. The review-then-continue approach:
- Prevents cascading rebases
- Catches issues early before building on them
- Avoids merge conflicts entirely
- Results in cleaner git history
