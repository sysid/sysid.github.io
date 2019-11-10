(window.webpackJsonp=window.webpackJsonp||[]).push([[6],{55:function(n,a,t){"use strict";t.r(a),t.d(a,"readingTime",function(){return u}),t.d(a,"default",function(){return h}),t.d(a,"tableOfContents",function(){return b}),t.d(a,"frontMatter",function(){return d});var s=t(14),e=(t(0),t(20)),o=t(56),p=t.n(o),i=t(57),c=t.n(i),l=t(58),r=t.n(l),u={text:"5 min read",minutes:4.45,time:267e3,words:890},k={},m="wrapper";function h(n){var a=n.components,t=Object(s.a)(n,["components"]);return Object(e.b)(m,Object.assign({},k,t,{components:a,mdxType:"MDXLayout"}),Object(e.b)("h1",{id:"fishy-affine-transformation"},"Fishy Affine Transformation"),Object(e.b)("p",null,"While working on the kaggle competition ",Object(e.b)("a",Object.assign({parentName:"p"},{href:"https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring"}),"https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring")," I hit the point when I wanted\nto align fish based on an annotation at the fish\u2019s head and tail, so that the fish is centered in the image, always in the same orientation\nand distracting picture information is minimized. This required:"),Object(e.b)("ol",null,Object(e.b)("li",{parentName:"ol"},"finding the fish (thanks Nathaniel Shimoni for annotating)"),Object(e.b)("li",{parentName:"ol"},"centering"),Object(e.b)("li",{parentName:"ol"},"rotatating"),Object(e.b)("li",{parentName:"ol"},"cropping")),Object(e.b)("p",null,"Mathematically the challenge is to find the associated  Affine Transformation. After years of working in a managerial role my linear algebra skills are a bit rusty so I decided to\ninvest the weekend."),Object(e.b)("h3",{id:"affine-transformation"},"Affine Transformation"),Object(e.b)("p",null,Object(e.b)("a",Object.assign({parentName:"p"},{href:"http://mathworld.wolfram.com/AffineTransformation.html"}),"Wolfram"),": An affine transformation is any transformation that preserves collinearity (i.e., all points lying on a line initially still lie on a line after transformation) and ratios of distances (e.g., the midpoint of a line segment remains the midpoint after transformation)."),Object(e.b)("p",null,"I decided to use ",Object(e.b)("a",Object.assign({parentName:"p"},{href:"http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#transformations"}),"CV2")," after hitting the wall with several other tools.\nIt was not the most convenient choice, but eventually it got me there. CV2 uses (2x3) transformation matrices for affine transformations so I had to adjust my 2d vectors accordingly."),Object(e.b)("p",null,"The reason: Homogeneous Coordinates."),Object(e.b)("p",null,"To combine rotation and translation in one operation one extra dimension is needed more than the model requires.\nFor planar things this is 3 components and for spatial things this is 4 components.\nThe operators take 3 components and return 3 components requiring 3x3 matrices."),Object(e.b)("p",null,"Using vector algebra with numpy requires some extra consideration but is possible. Basically a (2,) matrix represented the 2-dim vectors. Small letters\ndenoted vector variables and caps matrices."),Object(e.b)("h2",{id:"1-finding-the-fish"},"1. Finding the Fish"),Object(e.b)("p",null,"I used the annotations from labels produced by Nathaniel Shimoni and published on Kaggle (thanks for the great work!)."),Object(e.b)("p",null,"Using only fish with head and tail annotated, it was possible to get the vector representation of a fish as:"),Object(e.b)("pre",null,Object(e.b)("code",Object.assign({parentName:"pre"},{className:"language-python","data-language":"python","data-highlighted-line-numbers":"",dangerouslySetInnerHTML:{__html:'p_heads <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">(</span>img_data<span class="token punctuation">[</span><span class="token string">\'annotations\'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">\'x\'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> img_data<span class="token punctuation">[</span><span class="token string">\'annotations\'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">\'y\'</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span>\np_tails <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">(</span>img_data<span class="token punctuation">[</span><span class="token string">\'annotations\'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">\'x\'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> img_data<span class="token punctuation">[</span><span class="token string">\'annotations\'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">\'y\'</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span>\np_middle <span class="token operator">=</span> <span class="token punctuation">(</span>p_heads <span class="token operator">+</span> p_tails<span class="token punctuation">)</span><span class="token operator">/</span><span class="token number">2</span>\nv_fish <span class="token operator">=</span> p_heads <span class="token operator">-</span> p_tails\n'}}))),Object(e.b)("h2",{id:"2-centering"},"2. Centering"),Object(e.b)("p",null,"Centering fish is a basic translation in the 2-dim space."),Object(e.b)("pre",null,Object(e.b)("code",Object.assign({parentName:"pre"},{className:"language-python","data-language":"python","data-highlighted-line-numbers":"",dangerouslySetInnerHTML:{__html:'    <span class="token comment"># translate to center of img</span>\n    img_center <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">[</span>img_height<span class="token operator">/</span><span class="token number">2</span><span class="token punctuation">,</span> img_width<span class="token operator">/</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">)</span>\n    t <span class="token operator">=</span> img_center <span class="token operator">-</span> p_middle  <span class="token comment"># translation vector</span>\n    t <span class="token operator">=</span> np<span class="token punctuation">.</span>reshape<span class="token punctuation">(</span>t<span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>  <span class="token comment"># generate the 2x3 affine transformation matrix</span>\n    T <span class="token operator">=</span> np<span class="token punctuation">.</span>concatenate<span class="token punctuation">(</span><span class="token punctuation">(</span>np<span class="token punctuation">.</span>identity<span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">)</span><span class="token punctuation">,</span> t<span class="token punctuation">)</span><span class="token punctuation">,</span> axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span>\n'}}))),Object(e.b)("p",null,"The respective transformation matrix is:"),Object(e.b)("div",{className:r.a.logo},Object(e.b)("img",{src:p.a,className:r.a["logo-plant"],alt:"fishy-affine-transformation-translation"})),Object(e.b)("h2",{id:"3-rotating"},"3. Rotating"),Object(e.b)("p",null,"First I needed to find the angle for rotation. I wanted to have the fish oriented parallel to the x-axis with the head always being on the right. The dot-product of two vectors provides the\nangle in between, so I had to \u201cdot-product\u201d my fish vector with the x-axis:"),Object(e.b)("pre",null,Object(e.b)("code",Object.assign({parentName:"pre"},{className:"language-python","data-language":"python","data-highlighted-line-numbers":"",dangerouslySetInnerHTML:{__html:'<span class="token keyword">def</span> <span class="token function">unit_vector</span><span class="token punctuation">(</span>vector<span class="token punctuation">)</span><span class="token punctuation">:</span>\n    <span class="token triple-quoted-string string">""" Returns the unit vector of the vector."""</span>\n    <span class="token keyword">return</span> vector <span class="token operator">/</span> np<span class="token punctuation">.</span>linalg<span class="token punctuation">.</span>norm<span class="token punctuation">(</span>vector<span class="token punctuation">)</span>\n\n<span class="token keyword">def</span> <span class="token function">angle_between</span><span class="token punctuation">(</span>v1<span class="token punctuation">,</span> v2<span class="token punctuation">)</span><span class="token punctuation">:</span>\n    <span class="token triple-quoted-string string">""" Returns the angle in radians between vectors \'v1\' and \'v2\'::\n\n            >>> angle_between((1, 0, 0), (0, 1, 0))\n            1.5707963267948966\n            >>> angle_between((1, 0, 0), (1, 0, 0))\n            0.0\n            >>> angle_between((1, 0, 0), (-1, 0, 0))\n            3.141592653589793\n    """</span>\n    v1_u <span class="token operator">=</span> unit_vector<span class="token punctuation">(</span>v1<span class="token punctuation">)</span>\n    v2_u <span class="token operator">=</span> unit_vector<span class="token punctuation">(</span>v2<span class="token punctuation">)</span>\n    <span class="token keyword">return</span> np<span class="token punctuation">.</span>arccos<span class="token punctuation">(</span>np<span class="token punctuation">.</span>clip<span class="token punctuation">(</span>np<span class="token punctuation">.</span>dot<span class="token punctuation">(</span>v1_u<span class="token punctuation">,</span> v2_u<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">1.0</span><span class="token punctuation">,</span> <span class="token number">1.0</span><span class="token punctuation">)</span><span class="token punctuation">)</span>\n\nangle <span class="token operator">=</span> np<span class="token punctuation">.</span>rad2deg<span class="token punctuation">(</span>angle_between<span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">,</span> v_fish<span class="token punctuation">)</span><span class="token punctuation">)</span>\n'}}))),Object(e.b)("p",null,"Conveniently CV2 provides a function to find the necessary transformation matrix (cv2.getRotationMatrix2D)."),Object(e.b)("p",null,"A challenge was to find out, that the rotation angle returned always is between 0\u2013180\xb0, so the following conditional differentiation was necessary\n(rotation counter clockwise vs clockwise). It basically differentiates between the case that the head is above or below the tail:"),Object(e.b)("pre",null,Object(e.b)("code",Object.assign({parentName:"pre"},{className:"language-python","data-language":"python","data-highlighted-line-numbers":"",dangerouslySetInnerHTML:{__html:'    <span class="token comment"># get the Affine transformation matrix</span>\n    <span class="token keyword">if</span> p_heads<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span> <span class="token operator">></span> p_tails<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">:</span>  <span class="token comment"># head is above tail</span>\n        M <span class="token operator">=</span> cv2<span class="token punctuation">.</span>getRotationMatrix2D<span class="token punctuation">(</span><span class="token punctuation">(</span>p_middle<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> p_middle<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> angle<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>\n    <span class="token keyword">else</span><span class="token punctuation">:</span>\n        M <span class="token operator">=</span> cv2<span class="token punctuation">.</span>getRotationMatrix2D<span class="token punctuation">(</span><span class="token punctuation">(</span>p_middle<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> p_middle<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token operator">-</span>angle<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>\n'}}))),Object(e.b)("h2",{id:"putting-it-all-together"},"Putting it all together"),Object(e.b)("p",null,"Getting the resulting transformation from a translation and rotation mathematically translates to a matrix product and applying the resulting\ntransformation matrix to the fish vector. To make the multiplication of a 2x3 tranlation matrix and a 2x3 rotation matrix possible the\nfollowing steps were necesary (combination of two affine transformations):"),Object(e.b)("ul",null,Object(e.b)("li",{parentName:"ul"},"allocate A1, A2, R matrices, all 3x3 identity matrices (eyes)"),Object(e.b)("li",{parentName:"ul"},"replace the top part of A1 and A2 with the transformation matrices T and M"),Object(e.b)("li",{parentName:"ul"},"get the resulting transformation (matrix product)"),Object(e.b)("li",{parentName:"ul"},"return the first two rows of R")),Object(e.b)("p",null,"So RR was my final transformation matrix."),Object(e.b)("pre",null,Object(e.b)("code",Object.assign({parentName:"pre"},{className:"language-python","data-language":"python","data-highlighted-line-numbers":"",dangerouslySetInnerHTML:{__html:'    <span class="token comment"># compinte affine transform: make them 3x3</span>\n    <span class="token comment"># http://stackoverflow.com/questions/13557066/built-in-function-to-combine-affine-transforms-in-opencv</span>\n    A1 <span class="token operator">=</span> np<span class="token punctuation">.</span>identity<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span>\n    A2 <span class="token operator">=</span> np<span class="token punctuation">.</span>identity<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span>\n    R <span class="token operator">=</span> np<span class="token punctuation">.</span>identity<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span>\n    A1<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">2</span><span class="token punctuation">]</span> <span class="token operator">=</span> T\n    A2<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">2</span><span class="token punctuation">]</span> <span class="token operator">=</span> M\n    R <span class="token operator">=</span> A1@A2\n    RR <span class="token operator">=</span> R<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">2</span><span class="token punctuation">]</span>\n'}}))),Object(e.b)("p",null,"Getting the transformed image is now straightforward:"),Object(e.b)("pre",null,Object(e.b)("code",Object.assign({parentName:"pre"},{className:"language-python","data-language":"python","data-highlighted-line-numbers":"",dangerouslySetInnerHTML:{__html:'    dst <span class="token operator">=</span> cv2<span class="token punctuation">.</span>warpAffine<span class="token punctuation">(</span>img<span class="token punctuation">,</span> RR<span class="token punctuation">,</span> <span class="token punctuation">(</span>img_height<span class="token punctuation">,</span> img_width<span class="token punctuation">)</span><span class="token punctuation">)</span>\n'}}))),Object(e.b)("p",null,"The nice thing with this approach is that once you have got the final transformation matrix, all other points of interest can be transformed by this matrix,\ne.g. the head and tail annotations are transformed by the same matrix."),Object(e.b)("h2",{id:"result"},"Result"),Object(e.b)("p",null,"The blue point marks the head and the red point the tail. You can see the fish positioned arbitrarily in the image.\nWith the Affine Transformation the fish will be extracted and aligned.\nThe result is being displayed in the left upper corner."),Object(e.b)("div",{className:r.a.logo},Object(e.b)("img",{src:c.a,className:r.a["logo-plant"],alt:"fishy-affine-transformation-result"})),Object(e.b)("p",null,"With this technique I was able to align my fish and feed it into my machine learning models."),Object(e.b)("p",null,"Thanks for reading."),Object(e.b)("h5",{id:"disclaimer"},"Disclaimer"),Object(e.b)("p",null,"I use ",Object(e.b)("a",Object.assign({parentName:"p"},{href:"http://stackoverflow.com/"}),"http://stackoverflow.com/")," a lot. Not every source is quoted properly.",Object(e.b)("br",{parentName:"p"}),"\n","Other sources:",Object(e.b)("br",{parentName:"p"}),"\n",Object(e.b)("a",Object.assign({parentName:"p"},{href:"https://www.kaggle.com/qiubit/the-nature-conservancy-fisheries-monitoring/crop-fish"}),"https://www.kaggle.com/qiubit/the-nature-conservancy-fisheries-monitoring/crop-fish")))}h.isMDXComponent=!0;var b=function(){arguments.length>0&&void 0!==arguments[0]&&arguments[0];return[{id:"affine-transformation",level:3,title:"Affine Transformation",children:[]},{id:"1-finding-the-fish",level:2,title:"1. Finding the Fish",children:[]},{id:"2-centering",level:2,title:"2. Centering",children:[]},{id:"3-rotating",level:2,title:"3. Rotating",children:[]},{id:"putting-it-all-together",level:2,title:"Putting it all together",children:[]},{id:"result",level:2,title:"Result",children:[]}]},d={}},56:function(n,a,t){n.exports=t.p+"static/media/fishy-affine-transformation-translation.79a3cb79.png"},57:function(n,a,t){n.exports=t.p+"static/media/fishy-affine-transformation-result.bfb7e795.png"},58:function(n,a,t){n.exports={logo:"document_logo__2nJ5O","logo-navi":"document_logo-navi__2ffOv","logo-react":"document_logo-react__3fZrH","logo-plant":"document_logo-plant__EtAb-","Index-logo-react-spin":"document_Index-logo-react-spin__11FLa","Index-logo-navi-spin":"document_Index-logo-navi-spin__2JHkR"}}}]);
//# sourceMappingURL=6.90a8ff57.chunk.js.map