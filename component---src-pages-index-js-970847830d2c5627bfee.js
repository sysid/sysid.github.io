(window.webpackJsonp=window.webpackJsonp||[]).push([[4],{"0mN4":function(e,t,a){"use strict";a("OGtf")("fixed",(function(e){return function(){return e(this,"tt","","")}}))},"6Gk8":function(e,t,a){"use strict";a("f3/d"),a("0mN4");var r=a("pnhg"),i=a("q1tI"),n=a.n(i),s=a("9eSz"),d=a.n(s),o=a("p3AD");t.a=function(){var e=r.data,t=e.site.siteMetadata.author;return n.a.createElement("div",{style:{display:"flex",marginBottom:Object(o.a)(2.5)}},n.a.createElement(d.a,{fixed:e.avatar.childImageSharp.fixed,alt:t.name,style:{marginRight:Object(o.a)(.5),marginBottom:0,minWidth:50,borderRadius:"100%"},imgStyle:{borderRadius:"50%"}}),n.a.createElement("p",null,t.summary,", by"," ",n.a.createElement("a",{href:"mail://sysid@gmx.de"},"sysid"),"."))}},"9eSz":function(e,t,a){"use strict";a("rGqo"),a("yt8O"),a("Btvt"),a("XfO3"),a("EK0E"),a("INYr"),a("0mN4");var r=a("TqRt");t.__esModule=!0,t.default=void 0;var i,n=r(a("PJYZ")),s=r(a("VbXa")),d=r(a("8OQS")),o=r(a("pVnL")),l=r(a("q1tI")),c=r(a("17x9")),u=function(e){var t=(0,o.default)({},e),a=t.resolutions,r=t.sizes,i=t.critical;return a&&(t.fixed=a,delete t.resolutions),r&&(t.fluid=r,delete t.sizes),i&&(t.loading="eager"),t.fluid&&(t.fluid=w([].concat(t.fluid))),t.fixed&&(t.fixed=w([].concat(t.fixed))),t},f=function(e){var t=e.media;return!!t&&(b&&!!window.matchMedia(t).matches)},g=function(e){var t=e.fluid,a=e.fixed;return m(t||a).src},m=function(e){if(b&&function(e){return!!e&&Array.isArray(e)&&e.some((function(e){return void 0!==e.media}))}(e)){var t=e.findIndex(f);if(-1!==t)return e[t]}return e[0]},p=Object.create({}),h=function(e){var t=u(e),a=g(t);return p[a]||!1},y="undefined"!=typeof HTMLImageElement&&"loading"in HTMLImageElement.prototype,b="undefined"!=typeof window,S=b&&window.IntersectionObserver,v=new WeakMap;function E(e){return e.map((function(e){var t=e.src,a=e.srcSet,r=e.srcSetWebp,i=e.media,n=e.sizes;return l.default.createElement(l.default.Fragment,{key:t},r&&l.default.createElement("source",{type:"image/webp",media:i,srcSet:r,sizes:n}),l.default.createElement("source",{media:i,srcSet:a,sizes:n}))}))}function w(e){var t=[],a=[];return e.forEach((function(e){return(e.media?t:a).push(e)})),[].concat(t,a)}function O(e){return e.map((function(e){var t=e.src,a=e.media,r=e.tracedSVG;return l.default.createElement("source",{key:t,media:a,srcSet:r})}))}function R(e){return e.map((function(e){var t=e.src,a=e.media,r=e.base64;return l.default.createElement("source",{key:t,media:a,srcSet:r})}))}function I(e,t){var a=e.srcSet,r=e.srcSetWebp,i=e.media,n=e.sizes;return"<source "+(t?"type='image/webp' ":"")+(i?'media="'+i+'" ':"")+'srcset="'+(t?r:a)+'" '+(n?'sizes="'+n+'" ':"")+"/>"}var L=function(e,t){var a=(void 0===i&&"undefined"!=typeof window&&window.IntersectionObserver&&(i=new window.IntersectionObserver((function(e){e.forEach((function(e){if(v.has(e.target)){var t=v.get(e.target);(e.isIntersecting||e.intersectionRatio>0)&&(i.unobserve(e.target),v.delete(e.target),t())}}))}),{rootMargin:"200px"})),i);return a&&(a.observe(e),v.set(e,t)),function(){a.unobserve(e),v.delete(e)}},x=function(e){var t=e.src?'src="'+e.src+'" ':'src="" ',a=e.sizes?'sizes="'+e.sizes+'" ':"",r=e.srcSet?'srcset="'+e.srcSet+'" ':"",i=e.title?'title="'+e.title+'" ':"",n=e.alt?'alt="'+e.alt+'" ':'alt="" ',s=e.width?'width="'+e.width+'" ':"",d=e.height?'height="'+e.height+'" ':"",o=e.crossOrigin?'crossorigin="'+e.crossOrigin+'" ':"",l=e.loading?'loading="'+e.loading+'" ':"",c=e.draggable?'draggable="'+e.draggable+'" ':"";return"<picture>"+e.imageVariants.map((function(e){return(e.srcSetWebp?I(e,!0):"")+I(e)})).join("")+"<img "+l+s+d+a+r+t+n+i+o+c+'style="position:absolute;top:0;left:0;opacity:1;width:100%;height:100%;object-fit:cover;object-position:center"/></picture>'},A=function(e){var t=e.src,a=e.imageVariants,r=e.generateSources,i=e.spreadProps,n=e.ariaHidden,s=l.default.createElement(k,(0,o.default)({src:t},i,{ariaHidden:n}));return a.length>1?l.default.createElement("picture",null,r(a),s):s},k=l.default.forwardRef((function(e,t){var a=e.sizes,r=e.srcSet,i=e.src,n=e.style,s=e.onLoad,c=e.onError,u=e.loading,f=e.draggable,g=e.ariaHidden,m=(0,d.default)(e,["sizes","srcSet","src","style","onLoad","onError","loading","draggable","ariaHidden"]);return l.default.createElement("img",(0,o.default)({"aria-hidden":g,sizes:a,srcSet:r,src:i},m,{onLoad:s,onError:c,ref:t,loading:u,draggable:f,style:(0,o.default)({position:"absolute",top:0,left:0,width:"100%",height:"100%",objectFit:"cover",objectPosition:"center"},n)}))}));k.propTypes={style:c.default.object,onError:c.default.func,onLoad:c.default.func};var C=function(e){function t(t){var a;(a=e.call(this,t)||this).seenBefore=b&&h(t),a.isCritical="eager"===t.loading||t.critical,a.addNoScript=!(a.isCritical&&!t.fadeIn),a.useIOSupport=!y&&S&&!a.isCritical&&!a.seenBefore;var r=a.isCritical||b&&(y||!a.useIOSupport);return a.state={isVisible:r,imgLoaded:!1,imgCached:!1,fadeIn:!a.seenBefore&&t.fadeIn},a.imageRef=l.default.createRef(),a.handleImageLoaded=a.handleImageLoaded.bind((0,n.default)(a)),a.handleRef=a.handleRef.bind((0,n.default)(a)),a}(0,s.default)(t,e);var a=t.prototype;return a.componentDidMount=function(){if(this.state.isVisible&&"function"==typeof this.props.onStartLoad&&this.props.onStartLoad({wasCached:h(this.props)}),this.isCritical){var e=this.imageRef.current;e&&e.complete&&this.handleImageLoaded()}},a.componentWillUnmount=function(){this.cleanUpListeners&&this.cleanUpListeners()},a.handleRef=function(e){var t=this;this.useIOSupport&&e&&(this.cleanUpListeners=L(e,(function(){var e=h(t.props);t.state.isVisible||"function"!=typeof t.props.onStartLoad||t.props.onStartLoad({wasCached:e}),t.setState({isVisible:!0},(function(){return t.setState({imgLoaded:e,imgCached:!!t.imageRef.current.currentSrc})}))})))},a.handleImageLoaded=function(){var e,t,a;e=this.props,t=u(e),a=g(t),p[a]=!0,this.setState({imgLoaded:!0}),this.props.onLoad&&this.props.onLoad()},a.render=function(){var e=u(this.props),t=e.title,a=e.alt,r=e.className,i=e.style,n=void 0===i?{}:i,s=e.imgStyle,d=void 0===s?{}:s,c=e.placeholderStyle,f=void 0===c?{}:c,g=e.placeholderClassName,p=e.fluid,h=e.fixed,y=e.backgroundColor,b=e.durationFadeIn,S=e.Tag,v=e.itemProp,w=e.loading,I=e.draggable,L=!1===this.state.fadeIn||this.state.imgLoaded,C=!0===this.state.fadeIn&&!this.state.imgCached,V=(0,o.default)({opacity:L?1:0,transition:C?"opacity "+b+"ms":"none"},d),z="boolean"==typeof y?"lightgray":y,N={transitionDelay:b+"ms"},T=(0,o.default)({opacity:this.state.imgLoaded?0:1},C&&N,{},d,{},f),P={title:t,alt:this.state.isVisible?"":a,style:T,className:g,itemProp:v};if(p){var q=p,F=m(p);return l.default.createElement(S,{className:(r||"")+" gatsby-image-wrapper",style:(0,o.default)({position:"relative",overflow:"hidden"},n),ref:this.handleRef,key:"fluid-"+JSON.stringify(F.srcSet)},l.default.createElement(S,{"aria-hidden":!0,style:{width:"100%",paddingBottom:100/F.aspectRatio+"%"}}),z&&l.default.createElement(S,{"aria-hidden":!0,title:t,style:(0,o.default)({backgroundColor:z,position:"absolute",top:0,bottom:0,opacity:this.state.imgLoaded?0:1,right:0,left:0},C&&N)}),F.base64&&l.default.createElement(A,{ariaHidden:!0,src:F.base64,spreadProps:P,imageVariants:q,generateSources:R}),F.tracedSVG&&l.default.createElement(A,{ariaHidden:!0,src:F.tracedSVG,spreadProps:P,imageVariants:q,generateSources:O}),this.state.isVisible&&l.default.createElement("picture",null,E(q),l.default.createElement(k,{alt:a,title:t,sizes:F.sizes,src:F.src,crossOrigin:this.props.crossOrigin,srcSet:F.srcSet,style:V,ref:this.imageRef,onLoad:this.handleImageLoaded,onError:this.props.onError,itemProp:v,loading:w,draggable:I})),this.addNoScript&&l.default.createElement("noscript",{dangerouslySetInnerHTML:{__html:x((0,o.default)({alt:a,title:t,loading:w},F,{imageVariants:q}))}}))}if(h){var B=h,H=m(h),j=(0,o.default)({position:"relative",overflow:"hidden",display:"inline-block",width:H.width,height:H.height},n);return"inherit"===n.display&&delete j.display,l.default.createElement(S,{className:(r||"")+" gatsby-image-wrapper",style:j,ref:this.handleRef,key:"fixed-"+JSON.stringify(H.srcSet)},z&&l.default.createElement(S,{"aria-hidden":!0,title:t,style:(0,o.default)({backgroundColor:z,width:H.width,opacity:this.state.imgLoaded?0:1,height:H.height},C&&N)}),H.base64&&l.default.createElement(A,{ariaHidden:!0,src:H.base64,spreadProps:P,imageVariants:B,generateSources:R}),H.tracedSVG&&l.default.createElement(A,{ariaHidden:!0,src:H.tracedSVG,spreadProps:P,imageVariants:B,generateSources:O}),this.state.isVisible&&l.default.createElement("picture",null,E(B),l.default.createElement(k,{alt:a,title:t,width:H.width,height:H.height,sizes:H.sizes,src:H.src,crossOrigin:this.props.crossOrigin,srcSet:H.srcSet,style:V,ref:this.imageRef,onLoad:this.handleImageLoaded,onError:this.props.onError,itemProp:v,loading:w,draggable:I})),this.addNoScript&&l.default.createElement("noscript",{dangerouslySetInnerHTML:{__html:x((0,o.default)({alt:a,title:t,loading:w},H,{imageVariants:B}))}}))}return null},t}(l.default.Component);C.defaultProps={fadeIn:!0,durationFadeIn:500,alt:"",Tag:"div",loading:"lazy"};var V=c.default.shape({width:c.default.number.isRequired,height:c.default.number.isRequired,src:c.default.string.isRequired,srcSet:c.default.string.isRequired,base64:c.default.string,tracedSVG:c.default.string,srcWebp:c.default.string,srcSetWebp:c.default.string,media:c.default.string}),z=c.default.shape({aspectRatio:c.default.number.isRequired,src:c.default.string.isRequired,srcSet:c.default.string.isRequired,sizes:c.default.string.isRequired,base64:c.default.string,tracedSVG:c.default.string,srcWebp:c.default.string,srcSetWebp:c.default.string,media:c.default.string});C.propTypes={resolutions:V,sizes:z,fixed:c.default.oneOfType([V,c.default.arrayOf(V)]),fluid:c.default.oneOfType([z,c.default.arrayOf(z)]),fadeIn:c.default.bool,durationFadeIn:c.default.number,title:c.default.string,alt:c.default.string,className:c.default.oneOfType([c.default.string,c.default.object]),critical:c.default.bool,crossOrigin:c.default.oneOfType([c.default.string,c.default.bool]),style:c.default.object,imgStyle:c.default.object,placeholderStyle:c.default.object,placeholderClassName:c.default.string,backgroundColor:c.default.oneOfType([c.default.string,c.default.bool]),onLoad:c.default.func,onError:c.default.func,onStartLoad:c.default.func,Tag:c.default.string,itemProp:c.default.string,loading:c.default.oneOf(["auto","lazy","eager"]),draggable:c.default.bool};var N=C;t.default=N},INYr:function(e,t,a){"use strict";var r=a("XKFU"),i=a("CkkT")(6),n="findIndex",s=!0;n in[]&&Array(1)[n]((function(){s=!1})),r(r.P+r.F*s,"Array",{findIndex:function(e){return i(this,e,arguments.length>1?arguments[1]:void 0)}}),a("nGyu")(n)},OGtf:function(e,t,a){var r=a("XKFU"),i=a("eeVq"),n=a("vhPU"),s=/"/g,d=function(e,t,a,r){var i=String(n(e)),d="<"+t;return""!==a&&(d+=" "+a+'="'+String(r).replace(s,"&quot;")+'"'),d+">"+i+"</"+t+">"};e.exports=function(e,t){var a={};a[e]=t(d),r(r.P+r.F*i((function(){var t=""[e]('"');return t!==t.toLowerCase()||t.split('"').length>3})),"String",a)}},RXBc:function(e,t,a){"use strict";a.r(t),a.d(t,"pageQuery",(function(){return c}));var r=a("q1tI"),i=a.n(r),n=a("Wbzz"),s=a("6Gk8"),d=a("Bl7J"),o=a("vrFN"),l=a("p3AD");t.default=function(e){var t=e.data,a=e.location,r=t.site.siteMetadata.title,c=t.allMdx.edges;return i.a.createElement(d.a,{location:a,title:r},i.a.createElement(o.a,{title:"All posts"}),i.a.createElement(s.a,null),c.map((function(e){var t=e.node,a=t.frontmatter.title||t.fields.slug;return i.a.createElement("article",{key:t.fields.slug},i.a.createElement("header",null,i.a.createElement("h3",{style:{marginBottom:Object(l.a)(.25)}},i.a.createElement(n.Link,{style:{boxShadow:"none"},to:t.fields.slug},a)),i.a.createElement("small",null,t.frontmatter.date)),i.a.createElement("section",null,i.a.createElement("p",{dangerouslySetInnerHTML:{__html:t.frontmatter.description||t.excerpt}})))})))};var c="2167313165"},pnhg:function(e){e.exports=JSON.parse('{"data":{"avatar":{"childImageSharp":{"fixed":{"base64":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAACXBIWXMAAAXbAAAF2wGkowvQAAADUklEQVQ4y2VUSSi+Xxh9y5ysZEEpZSksTSXCkmRMkqREhFhIsmAhU8qYFLIQkaLYCInMQxaKImXOlGQevvf83vP439v3+Z+6dd97n+fcZzjPa5imCcJms8kiXl5e0N3djYGBAXR0dMDFxQWGYaChoQFdXV1oa2vD8/Oz2NL/5+dH7w1FpnB+fo7IyEgh8fHxEaK6ujoh4d7Pzw/u7u6IiIjAxcWF9lMchtqcnp6KEZ24+P3x8YGFhQXttLW1JY4kUnahoaE4ODiQe0ZqfH9/y0dubi78/f2xvLyM1dVV2IM2yk5hcXERa2trCA4ORkJCgmPKTU1N8lp+fr52ULX5/PzUZ19fX3Km6k6wHPQtLy//TXlyclIO2tvbcXR0JCnRSRWaYOpcCuqettfX15iYmBCOoaEhGIODg3Bzc9NdoyEjIW5vb1FbW4uSkhIUFxejqqoKZ2dnDtEST09P8PT0RF9fH4y8vDw4Ozvr4iujx8dHZGZmYnp6Gq+vr7Lm5+eRnp6uu6saurS0BA8PD7kzGOrw8LBDeru7u0hLS8Ps7Cz+YnNzE8nJydjY2JBHFBiQdD4+Ph4BAQHY2dmRqJhedXW1foRNYSRshKrj+Pg4ampqUFRUhMvLS5ycnCAoKOhXdlNTU8J8fHyMubk5FBQUOETEaGZmZrC3t4e4uDiHZlVWVmJ0dFT2Tk5OaG1thcEDEjJCNqaxsVFqRykRTO3m5kYKv7KyImccy4yMDNTX10s9t7e34erq+kvIbqWmpgopO0owBWrSfiTtUVhYKCT2Gk5JScH7+/uvsImYmBgZo6urKxm70tJSvL29CSmnxF5OFPHh4SHu7+9lSkJCQvRjevSYsq+vr7xWUVGBnp4eWX8xMjKClpYWNDc3i623t7eWHLkMe+0xmvDwcNETQY12dnbi7u4ODw8P6O/vR3Z2ttzx0cDAQC0d/bdRc6nSycrKQlJSkjZiXb28vES4ZWVlerZZx9jYWB0ZeeTnYP4HK0oym1aDTKuDsuey5GRaqcmyRK3PLcmYUVFRsrceVjSmYT/wBOUQFhaG/f19rK+vo7e3V//7KClOCu8SExMRHR2Nvz9oEpLeZhHy1DY2NmZTEalllcCWk5Pzv3OraeJDX3Jw/w+1xNgvXFo/BQAAAABJRU5ErkJggg==","width":50,"height":50,"src":"/static/cb3e4801e88879c2218e379df90c276f/8ba1e/munggoggo.png","srcSet":"/static/cb3e4801e88879c2218e379df90c276f/8ba1e/munggoggo.png 1x,\\n/static/cb3e4801e88879c2218e379df90c276f/f937a/munggoggo.png 1.5x,\\n/static/cb3e4801e88879c2218e379df90c276f/71eb7/munggoggo.png 2x"}}},"site":{"siteMetadata":{"author":{"name":"sysid","summary":"always be building","email":"sysid@gmx.de"},"social":{"twitter":""}}}}}')}}]);
//# sourceMappingURL=component---src-pages-index-js-970847830d2c5627bfee.js.map