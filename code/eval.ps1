for ($i=1; $i -le 2; $i++) {
	python -u .\eval_embeddings.py | Tee-Object -FilePath .\out.log | Write-Host
	
	# echo $i > out.log
	
	$source_path = '..\results\weisfeiler_lehman\'
	# $mutag_path = $source_path + 'MUTAG.txt'
	
	# echo 'bla' > $mutag_path
	
	$target_path = $source_path + $i.ToString() + '. iteration'
	if(-not (Test-Path $target_path)) {
		New-Item -ItemType directory -Path $target_path
	}
		
	robocopy /mov $source_path $target_path
	
	robocopy /mov . $target_path out.log
}