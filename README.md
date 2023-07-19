# GDSC6

ls -R src | grep ':$' | sed -e 's/:$//' -e 's/[^\/]*\//|  /g' -e 's/|  \([^|]\)/|–– \1/g' 
