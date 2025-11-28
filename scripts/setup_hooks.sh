#!/bin/sh
#
# Script to install git pre-commit hook
#

HOOK_DIR=".git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

if [ ! -d "$HOOK_DIR" ]; then
    echo "Error: .git directory not found. Are you in the root of the repo?"
    exit 1
fi

echo "#!/bin/sh
#
# Pre-commit hook to run ruff linting
#

echo \"Running ruff linting...\"
ruff check .

if [ \$? -ne 0 ]; then
    echo \"Linting failed. Please fix the errors before committing.\"
    exit 1
fi

echo \"Linting passed.\"
" > "$HOOK_FILE"

chmod +x "$HOOK_FILE"
echo "Pre-commit hook installed successfully."
