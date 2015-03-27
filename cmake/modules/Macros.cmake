if ( APPLE )
   macro( EXT_GET NAME url out )
     set( ${NAME} curl ${url} -o ${out} )
   endmacro( EXT_GET )
else()
   macro( EXT_GET NAME url out )
     set( ${NAME} wget ${url} -O ${out} )
   endmacro( EXT_GET )
endif( APPLE) 

