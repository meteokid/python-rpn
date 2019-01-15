#!/bin/ksh

git tag -d original_clone_subtree

dirslist=""
for i in $(cat DEPENDENCIES 2>/dev/null | sed 's/ //g' | tr '\n' ' ') ; do \
   name=${i%%=*} ; rt=${i#*=} ; repos=${rt%@*} ; tag=${rt##*@}

   git remote remove ${name} 2>/dev/null || true

   ## Optionally remove subtree split-ted branches
   # bname=${name}-${tag}-${USER}-branch
   # git branch -D ${bname}

   dirslist="${dirslist} ${name}"
done

set -x
git filter-branch --index-filter "git rm --cached --ignore-unmatch -rf ${dirslist}" --prune-empty -f HEAD

rm -rf .git/refs/original/*
git reflog expire --all --expire-unreachable=0

git rebase -i original_clone

git gc --prune=all
## Needed?
# git repack -A -d
# git prune

# git push --forced
## Keep in mind, because of the rebase, if you want to push to the same branch you'll need to pass the option --force.
