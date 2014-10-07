<html lang="en" class="no-js one-page-layout" data-classic-layout="false" data-mobile-only-classic-layout="true" data-inanimation="fadeInUp" data-outanimation="fadeOutDownBig"><!-- InstanceBegin template="/Templates/layout.dwt" codeOutsideHTMLIsLocked="false" --><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="description" content="cvCard - Responsive HTML5 vCard Template">
<meta name="keywords" content="vcard, html5, portfolio">
<meta name="author" content="Pixelwars">

<link rel="stylesheet" href="static//style.css" type="text/css">
    
<link rel="icon" type="image/png" href="http://crcv.ucf.edu/people/phd_students/mahdi/images/favicon.png">


<div class="container700 vcard">


<h1>Thinking with Images</h1>
<br>
<h2>Image retrieval from multiple query images.</h2>
<br>
%if count == 0:
	<h3>Please load images to be used as queries</h3>
%else:
	<table>
	<tr><th>Query</th><th>Closest Images in the dataset</th></tr>
%for row in range(len(querys)):
    <tr>
        <td><img width=220 height=220  src='static/queries/{{querys[row]}}' alt='query'  </td>
	<td><table>
	% for iOut in range(4):
	  <tr>
	 <td>  <img width=55 height=55  src='static/{{NNs[row][iOut*4]}}' alt='query'   </td>
	 <td>  <img width=55 height=55  src='static/{{NNs[row][iOut*4+1]}}' alt='query'   </td>
	 <td>  <img width=55 height=55  src='static/{{NNs[row][iOut*4+2]}}' alt='query'   </td>
	 <td>  <img width=55 height=55  src='static/{{NNs[row][iOut*4+3]}}' alt='query'   </td>
	 </tr>
	%end

	</table></td>

    </tr>
%end
	</table>
%end


<br>

<form action="/upload" method="post" enctype="multipart/form-data">
  Select a file: <input type="file" name="upload" />
  <input type="submit" value="Start upload" />
</form>

<form action="/clear" method="post" >
   <input type="submit" value="Start over" />
</form>

<form action="/search" method="post" >
   <input type="submit" value="search" />
</form>

<br>
<h2>	{{outlabel}}  </h2>

</div>
