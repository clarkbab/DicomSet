# Landmarks regexp

I don't know what level to apply this at, should the rtstruct dicom utils
handle regions/landmarks separately, i.e. split there, or do we split at
the mapped region level.

It could be useful, if landmarks have different names, to first map them
to a consistent name and then use a regexp to filter them. But, a regexp
could easily be expanded to match all types too - not much difference between
the two approaches.

I think we can apply the filtering at the rstruct dicom utils level, but
a regexp could be passed by the user which would be applied after region
mapping has been carried out. For now, just apply at the base level.

