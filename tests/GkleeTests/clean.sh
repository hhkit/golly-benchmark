 #!/bin/bash

echo "This will remove all files not under version control [git clean -fx], are you sure? [Y|n]"

read RES

if [[ "$RES" == "Y" ]]
then
    git clean -fx
    echo "Cleaned."
    exit 0
else
    echo "Nothing done."
    exit 1
fi

# find . -type f -not -name "*.cu" -not -name "*.sh" -not -name "*.py" -not -name "*.txt" -not -name "README*" -not -name "*.tgz"  | xargs rm -fv
# rm -vrf */klee*
