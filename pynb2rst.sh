curr_dir=$(pwd)
sphinx_dir_name="sphinx_setup"
docs_dir_name="docs"
sphinx_dir_path="$curr_dir/$sphinx_dir_name"
list_of_contents_in_curr_dir=$(ls)

# echo $list_of_contents_in_curr_dir
rm -rf "$sphinx_dir_path/"*"_files/"
for content in $list_of_contents_in_curr_dir
do
    if [ -d $content ];then
        if [[ $content == $sphinx_dir_name ]] || [[ $content == $docs_dir_name ]];then
            echo "*----------- Skipping for $content "
        else
            echo "*----------- Starting for $content ----------------*"
            jupyter-nbconvert "$curr_dir/$content/*.ipynb" --to rst
            mv -f "$curr_dir/$content/"*".rst"  $sphinx_dir_path
            mv -f "$curr_dir/$content/"*"_files/" $sphinx_dir_path
            echo "*----------- Completed for $content ----------------*"
        fi
    else
        echo "$content is not a directory"
    fi
done 

sphinx-build -b html $sphinx_dir_name $docs_dir_name


