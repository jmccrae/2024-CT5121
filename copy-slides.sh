cp lectures/*.md slides
sed -i 's/\\$\\$$/$$`/' slides/*.md
sed -i 's/^\\$\\$/`$$/' slides/*.md
 
