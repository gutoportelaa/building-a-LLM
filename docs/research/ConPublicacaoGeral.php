<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
            "http://www.w3.org/TR/1999/REC-html401-19991224/loose.dtd">
  <HTML DIR='LTR'>
  <HEAD>
   <TITLE></TITLE>
   <META http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
   <META http-equiv="Expires" content="Fri, Jan 01 1900 00:00:00 GMT"/>
   <META http-equiv="Last-Modified" content="Wed, 08 Apr 2026 14:34:35 GMT"/>
   <META http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate"/>
   <META http-equiv="Cache-Control" content="post-check=0, pre-check=0"/>
   <META http-equiv="Pragma" content="no-cache"/>
   <form name="form_ajax_redir_1" method="post" style="display: none">
    <input type="hidden" name="nmgp_parms">
    <input type="hidden" name="nmgp_outra_jan">
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75">
   </form>
   <form name="form_ajax_redir_2" method="post" style="display: none"> 
    <input type="hidden" name="nmgp_parms">
    <input type="hidden" name="nmgp_url_saida">
    <input type="hidden" name="script_case_init">
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75">
   </form>
   <script type="text/javascript" src="ConPublicacaoGeral_jquery.js"></script>
   <script type="text/javascript" src="ConPublicacaoGeral_ajax.js"></script>
   <script type="text/javascript">
     var sc_ajaxBg = '#6e6e6e';
     var sc_ajaxBordC = '#8DA0C8 ';
     var sc_ajaxBordS = 'solid';
     var sc_ajaxBordW = '1px';
   </script>
   <script type="text/javascript" src="/consulta/_lib/prod/third/jquery/js/jquery.js"></script>
   <script type="text/javascript" src="/consulta/_lib/prod/third/jquery/js/jquery-ui.js"></script>
   <link rel="stylesheet" href="/consulta/_lib/prod/third/jquery/css/smoothness/jquery-ui.css" type="text/css" media="screen" />
   <script type="text/javascript" src="/consulta/_lib/prod/third/jquery_plugin/touch_punch/jquery.ui.touch-punch.min.js"></script>
   <script type="text/javascript" src="/consulta/_lib/prod/third/jquery_plugin/malsup-blockui/jquery.blockUI.js"></script>
   <script type="text/javascript">var sc_pathToTB = '/consulta/_lib/prod/third/jquery_plugin/thickbox/';</script>
   <script type="text/javascript" src="/consulta/_lib/prod/third/jquery_plugin/thickbox/thickbox-compressed.js"></script>
   <script type="text/javascript" src="../_lib/lib/js/jquery.scInput.js"></script>
   <link rel="stylesheet" href="/consulta/_lib/prod/third/jquery_plugin/thickbox/thickbox.css" type="text/css" media="screen" />
   <link rel="stylesheet" type="text/css" href="../_lib/buttons/Scriptcase5_Silver/Scriptcase5_Silver.css" /> 
   <link rel="stylesheet" type="text/css" href="../_lib/css/Tipo2/Tipo2_form.css" /> 
   <link rel="stylesheet" type="text/css" href="../_lib/css/Tipo2/Tipo2_formLTR.css" /> 
   <link rel="stylesheet" type="text/css" href="../_lib/css/Tipo2/Tipo2_appdiv.css" /> 
   <link rel="stylesheet" type="text/css" href="../_lib/css/Tipo2/Tipo2_appdivLTR.css" /> 
   <style type="text/css">
     #quicksearchph_top {
       position: relative;
     }
     #quicksearchph_top img {
       position: absolute;
       top: 0;
       right: 0;
     }
   </style>
   <script type="text/javascript"> 
   var SC_Link_View = false;
   var Qsearch_ok = true;
   var scQSInit = true;
   var scQtReg  = 10;
  function scSetFixedHeaders() {
   var divScroll, gridHeaders, headerPlaceholder;
   gridHeaders = $(".sc-ui-grid-header-row-ConPublicacaoGeral-1");
   headerPlaceholder = $("#sc-id-fixedheaders-placeholder");
   scSetFixedHeadersContents(gridHeaders, headerPlaceholder);
   scSetFixedHeadersSize(gridHeaders);
   scSetFixedHeadersPosition(gridHeaders, headerPlaceholder);
   if (scIsHeaderVisible(gridHeaders)) {
    headerPlaceholder.hide();
   }
   else {
    headerPlaceholder.show();
   }
  }
  function scSetFixedHeadersPosition(gridHeaders, headerPlaceholder) {
   headerPlaceholder.css({"top": "0", "left": (Math.floor(gridHeaders.position().left) - $(document).scrollLeft()) + "px"});
  }
  function scIsHeaderVisible(gridHeaders) {
   return gridHeaders.offset().top > $(document).scrollTop();
  }
  function scSetFixedHeadersContents(gridHeaders, headerPlaceholder) {
   var i, htmlContent;
   htmlContent = "<table id=\"sc-id-fixed-headers\" class=\"scGridTabela\">";
   for (i = 0; i < gridHeaders.length; i++) {
    htmlContent += "<tr class=\"scGridLabel\" id=\"sc-id-fixed-headers-row-" + i + "\">" + $(gridHeaders[i]).html() + "</tr>";
   }
   htmlContent += "</table>";
   headerPlaceholder.html(htmlContent);
  }
  function scSetFixedHeadersSize(gridHeaders) {
   var i, j, headerColumns, gridColumns, cellHeight, cellWidth, tableOriginal, tableHeaders;
   tableOriginal = $("#sc-ui-grid-body-6898a72d");
   tableHeaders = document.getElementById("sc-id-fixed-headers");
   $(tableHeaders).css("width", $(tableOriginal).outerWidth());
   for (i = 0; i < gridHeaders.length; i++) {
    headerColumns = $("#sc-id-fixed-headers-row-" + i).find("td");
    gridColumns = $(gridHeaders[i]).find("td");
    for (j = 0; j < gridColumns.length; j++) {
     if (window.getComputedStyle(gridColumns[j])) {
      cellWidth = window.getComputedStyle(gridColumns[j]).width;
      cellHeight = window.getComputedStyle(gridColumns[j]).height;
     }
     else {
      cellWidth = $(gridColumns[j]).width() + "px";
      cellHeight = $(gridColumns[j]).height() + "px";
     }
     $(headerColumns[j]).css({
      "width": cellWidth,
      "height": cellHeight
     });
    }
   }
  }
  function SC_init_jquery(isScrollNav){ 
   $(function(){ 
     $('#SC_fast_search_top').keyup(function(e) {
       scQuickSearchKeyUp('top', e);
     });
     $('#id_F0_top').keyup(function(e) {
       var keyPressed = e.charCode || e.keyCode || e.which;
       if (13 == keyPressed) {
          return false; 
       }
     });
     $('#id_F0_bot').keyup(function(e) {
       var keyPressed = e.charCode || e.keyCode || e.which;
       if (13 == keyPressed) {
          return false; 
       }
     });
     $(".scBtnGrpText").mouseover(function() { $(this).addClass("scBtnGrpTextOver"); }).mouseout(function() { $(this).removeClass("scBtnGrpTextOver"); });
     $(".scBtnGrpClick").find("a").click(function(e){
        e.preventDefault();
     });
     $(".scBtnGrpClick").click(function(){
        var aObj = $(this).find("a"), aHref = aObj.attr("href");
        if ("javascript:" == aHref.substr(0, 11)) {
           eval(aHref.substr(11));
        }
        else {
           aObj.trigger("click");
        }
      }).mouseover(function(){
        $(this).css("cursor", "pointer");
     });
   }); 
  }
  SC_init_jquery(false);
   $(window).load(function() {
     scQuickSearchInit(false, '');
     $('#SC_fast_search_top').listen();
     scQuickSearchKeyUp('top', null);
     scQSInit = false;
   });
   function scQuickSearchSubmit_top() {
     document.F0_top.nmgp_opcao.value = 'fast_search';
     document.F0_top.submit();
   }
   function scQuickSearchInit(bPosOnly, sPos) {
     if (!bPosOnly) {
       if ('' == sPos || 'top' == sPos) scQuickSearchSize('SC_fast_search_top', 'SC_fast_search_close_top', 'SC_fast_search_submit_top', 'quicksearchph_top');
     }
   }
   function scQuickSearchSize(sIdInput, sIdClose, sIdSubmit, sPlaceHolder) {
     if($('#' + sIdInput).length)
     {
         var oInput = $('#' + sIdInput),
             oClose = $('#' + sIdClose),
             oSubmit = $('#' + sIdSubmit),
             oPlace = $('#' + sPlaceHolder),
             iInputP = parseInt(oInput.css('padding-right')) || 0,
             iInputB = parseInt(oInput.css('border-right-width')) || 0,
             iInputW = oInput.outerWidth(),
             iPlaceW = oPlace.outerWidth(),
             oInputO = oInput.offset(),
             oPlaceO = oPlace.offset(),
             iNewRight;
         iNewRight = (iPlaceW - iInputW) - (oInputO.left - oPlaceO.left) + iInputB + 1;
         oInput.css({
           'height': Math.max(oInput.height(), 16) + 'px',
           'padding-right': iInputP + 16 + 5 + 'px'
         });
         oClose.css({
           'right': iNewRight + 5 + 'px',
           'cursor': 'pointer'
         });
         oSubmit.css({
           'right': iNewRight + 5 + 'px',
           'cursor': 'pointer'
         });
     }
   }
   function scQuickSearchKeyUp(sPos, e) {
    if(typeof scQSInitVal !== 'undefined')
    {
     if ('' != scQSInitVal && $('#SC_fast_search_' + sPos).val() == scQSInitVal && scQSInit) {
       $('#SC_fast_search_close_' + sPos).show();
       $('#SC_fast_search_submit_' + sPos).hide();
     }
     else {
       $('#SC_fast_search_close_' + sPos).hide();
       $('#SC_fast_search_submit_' + sPos).show();
     }
     if (null != e) {
       var keyPressed = e.charCode || e.keyCode || e.which;
       if (13 == keyPressed) {
         if ('top' == sPos) nm_gp_submit_qsearch('top');
       }
     }
    }
   }
   function scBtnGroupByShow(sUrl, sPos) {
     $.ajax({
       type: "GET",
       dataType: "html",
       url: sUrl
     }).success(function(data) {
       $("#sc_id_groupby_placeholder_" + sPos).show();
       $("#sc_id_groupby_placeholder_" + sPos).find("td").html(data);
     });
   }
   function scBtnGroupByHide(sPos) {
     $("#sc_id_groupby_placeholder_" + sPos).hide();
     $("#sc_id_groupby_placeholder_" + sPos).find("td").html("");
   }
   function scBtnSaveGridShow(sUrl, sPos) {
     $.ajax({
       type: "GET",
       dataType: "html",
       url: sUrl
     }).success(function(data) {
       $("#sc_id_save_grid_placeholder_" + sPos).find("td").html(data);
       $("#sc_id_save_grid_placeholder_" + sPos).show();
     });
   }
   function scBtnSaveGridHide(sPos) {
     $("#sc_id_save_grid_placeholder_" + sPos).hide();
     $("#sc_id_save_grid_placeholder_" + sPos).find("td").html("");
   }
   function scBtnSelCamposShow(sUrl, sPos) {
     $.ajax({
       type: "GET",
       dataType: "html",
       url: sUrl
     }).success(function(data) {
       $("#sc_id_sel_campos_placeholder_" + sPos).find("td").html(data);
       $("#sc_id_sel_campos_placeholder_" + sPos).show();
     });
   }
   function scBtnSelCamposHide(sPos) {
     $("#sc_id_sel_campos_placeholder_" + sPos).hide();
     $("#sc_id_sel_campos_placeholder_" + sPos).find("td").html("");
   }
   function scBtnOrderCamposShow(sUrl, sPos) {
     $.ajax({
       type: "GET",
       dataType: "html",
       url: sUrl
     }).success(function(data) {
       $("#sc_id_order_campos_placeholder_" + sPos).find("td").html(data);
       $("#sc_id_order_campos_placeholder_" + sPos).show();
     });
   }
   function scBtnOrderCamposHide(sPos) {
     $("#sc_id_order_campos_placeholder_" + sPos).hide();
     $("#sc_id_order_campos_placeholder_" + sPos).find("td").html("");
   }
   var scBtnGrpStatus = {};
   function scBtnGrpShow(sGroup) {
     var btnPos = $('#sc_btgp_btn_' + sGroup).offset();
     scBtnGrpStatus[sGroup] = 'open';
     $('#sc_btgp_btn_' + sGroup).mouseout(function() {
       setTimeout(function() {
         scBtnGrpHide(sGroup);
       }, 1000);
     });
     $('#sc_btgp_div_' + sGroup + ' span a').click(function() {
       scBtnGrpStatus[sGroup] = 'out';
       scBtnGrpHide(sGroup);
     });
     $('#sc_btgp_div_' + sGroup).css({
       'left': btnPos.left
     })
     .mouseover(function() {
       scBtnGrpStatus[sGroup] = 'over';
     })
     .mouseleave(function() {
       scBtnGrpStatus[sGroup] = 'out';
       setTimeout(function() {
         scBtnGrpHide(sGroup);
       }, 1000);
     })
     .show('fast');
   }
   function scBtnGrpHide(sGroup) {
     if ('over' != scBtnGrpStatus[sGroup]) {
       $('#sc_btgp_div_' + sGroup).hide('fast');
     }
   }
   </script> 
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_grid.css" type="text/css" media="screen" />
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_gridLTR.css" type="text/css" media="screen" />
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_error.css" type="text/css" media="screen" />
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_errorLTR.css" type="text/css" media="screen" />
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_tab.css" type="text/css" media="screen" />
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_tabLTR.css" type="text/css" media="screen" />
  <style type="text/css">
  .css_iframes   { margin-bottom: 0px; margin-left: 0px;  margin-right: 0px;  margin-top: 0px; }
       .ttip {border:1px solid black;font-size:12px;layer-background-color:lightyellow;background-color:lightyellow}
  </style>
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_btngrp.css" type="text/css" media="screen" />
   <link rel="stylesheet" href="../_lib/css/Tipo2/Tipo2_btngrpLTR.css" type="text/css" media="screen" />
   <link rel="stylesheet" type="text/css" href="/consulta/ConPublicacaoGeral/ConPublicacaoGeral_grid_ltr.css" />
  </HEAD>
  <body class="scGridPage"  style="">
  
<div id="id_debug_window" style="display: none; position: absolute; left: 50px; top: 50px"><table class="scFormMessageTable">
<tr><td class="scFormMessageTitle"><a  title="Fechar" style="vertical-align: middle; display:inline-block;" onClick="nmAjaxHideDebug(); return false;"><img  src="/consulta/_lib/img/scriptcase__NM__nm_Scriptcase5_Silver_berrm_clse.gif" style="border-width: 0; cursor: pointer" /></a>
&nbsp;&nbsp;Output</td></tr>
<tr><td class="scFormMessageMessage" style="padding: 0px; vertical-align: top"><div style="padding: 2px; height: 200px; width: 350px; overflow: auto" id="id_debug_text"></div></td></tr>
</table></div>
      <div id="tooltip" style="position:absolute;visibility:hidden;border:1px solid black;font-size:12px;layer-background-color:lightyellow;background-color:lightyellow;padding:1px"></div>
   <form name="F3" method="post" 
                     action="ConPublicacaoGeral.php" 
                     target="_self" style="display: none"> 
    <input type="hidden" name="nmgp_chave" value=""/>
    <input type="hidden" name="nmgp_opcao" value=""/>
    <input type="hidden" name="nmgp_ordem" value=""/>
    <input type="hidden" name="SC_lig_apl_orig" value="ConPublicacaoGeral"/>
    <input type="hidden" name="nmgp_parm_acum" value=""/>
    <input type="hidden" name="nmgp_quant_linhas" value=""/>
    <input type="hidden" name="nmgp_url_saida" value=""/>
    <input type="hidden" name="nmgp_parms" value=""/>
    <input type="hidden" name="nmgp_tipo_pdf" value=""/>
    <input type="hidden" name="nmgp_outra_jan" value=""/>
    <input type="hidden" name="nmgp_orig_pesq" value=""/>
    <input type="hidden" name="script_case_init" value="7840"/> 
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75"/> 
   </form> 
   <form name="F4" method="post" 
                     action="ConPublicacaoGeral.php" 
                     target="_self" style="display: none"> 
    <input type="hidden" name="nmgp_opcao" value="rec"/>
    <input type="hidden" name="rec" value=""/>
    <input type="hidden" name="nm_call_php" value=""/>
    <input type="hidden" name="script_case_init" value="7840"/> 
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75"/> 
   </form> 
   <form name="F5" method="post" 
                     action="ConPublicacaoGeral_pesq.class.php" 
                     target="_self" style="display: none"> 
    <input type="hidden" name="script_case_init" value="7840"/> 
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75"/> 
   </form> 
   <form name="F6" method="post" 
                     action="ConPublicacaoGeral.php" 
                     target="_self" style="display: none"> 
    <input type="hidden" name="script_case_init" value="7840"/> 
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75"/> 
   </form> 
  <form name="Fdoc_word" method="post" 
        action="ConPublicacaoGeral.php" 
        target="_self"> 
    <input type="hidden" name="nmgp_opcao" value="doc_word"/> 
    <input type="hidden" name="nmgp_cor_word" value="AM"/> 
    <input type="hidden" name="nmgp_navegator_print" value=""/> 
    <input type="hidden" name="script_case_init" value="7840"/> 
    <input type="hidden" name="script_case_session" value="nuvh154aavroihs40r2786vf75"> 
  </form> 
   <script type="text/javascript">
    document.Fdoc_word.nmgp_navegator_print.value = navigator.appName;
   function nm_gp_word_conf(cor)
   {
       document.Fdoc_word.nmgp_cor_word.value = cor;
       document.Fdoc_word.submit();
   }
   var obj_tr      = "";
   var css_tr      = "";
   var field_over  = 1;
   var field_click = 1;
   function over_tr(obj, class_obj)
   {
       if (field_over != 1)
       {
           return;
       }
       if (obj_tr == obj)
       {
           return;
       }
       obj.className = 'scGridFieldOver';
   }
   function out_tr(obj, class_obj)
   {
       if (field_over != 1)
       {
           return;
       }
       if (obj_tr == obj)
       {
           return;
       }
       obj.className = class_obj;
   }
   function click_tr(obj, class_obj)
   {
       if (field_click != 1)
       {
           return;
       }
       if (obj_tr != "")
       {
           obj_tr.className = css_tr;
       }
       css_tr        = class_obj;
       if (obj_tr == obj)
       {
           obj_tr     = '';
           return;
       }
       obj_tr        = obj;
       css_tr        = class_obj;
       obj.className = 'scGridFieldClick';
   }
   function sc_btn_AgruparPDFs()
   {
       var vls_check = "", checked_records, i;
       checked_records = $(".sc-ui-check-run").filter(":checked");
       for (i = 0; i <= checked_records.length; i++)
       {
           vls_check += (vls_check != "") ? ";" : "";
           vls_check += $(checked_records[i]).val();
       }
       if (vls_check == "" || vls_check == "0" || vls_check == "undefined")
       {
           alert ("Selecionar dados");
           return;
       }
       document.FBtn_Run.nm_run_opt_sel.value = vls_check;
       document.FBtn_Run.target = "_self";
       document.FBtn_Run.nm_call_php.value = "AgruparPDFs";
       document.FBtn_Run.submit() ;
   }
   function nm_marca_check_grid(obj_mark)
   {
      $(".sc-ui-check-run").prop("checked", $(obj_mark).prop("checked"));
   }
   var tem_hint;
   function nm_mostra_hint(nm_obj, nm_evt, nm_mens)
   {
       if (nm_mens == "")
       {
           return;
       }
       tem_hint = true;
       if (document.layers)
       {
           theString="<DIV CLASS='ttip'>" + nm_mens + "</DIV>";
           document.tooltip.document.write(theString);
           document.tooltip.document.close();
           document.tooltip.left = nm_evt.pageX + 14;
           document.tooltip.top = nm_evt.pageY + 2;
           document.tooltip.visibility = "show";
       }
       else
       {
           if(document.getElementById)
           {
              nmdg_nav = navigator.appName;
              elm = document.getElementById("tooltip");
              elml = nm_obj;
              elm.innerHTML = nm_mens;
              if (nmdg_nav == "Netscape")
              {
                  elm.style.height = elml.style.height;
                  elm.style.top = nm_evt.pageY + 2 + 'px';
                  elm.style.left = nm_evt.pageX + 14 + 'px';
              }
              else
              {
                  elm.style.top = nm_evt.y + document.body.scrollTop + 10 + 'px';
                  elm.style.left = nm_evt.x + document.body.scrollLeft + 10 + 'px';
              }
              elm.style.visibility = "visible";
           }
       }
   }
   function nm_apaga_hint()
   {
       if (!tem_hint)
       {
           return;
       }
       tem_hint = false;
       if (document.layers)
       {
           document.tooltip.visibility = "hidden";
       }
       else
       {
           if(document.getElementById)
           {
              elm.style.visibility = "hidden";
           }
       }
   }
   nm_gp_ini = "ini";
   nm_gp_rec_ini = "0";
   nm_gp_rec_fim = "11";
   function nm_gp_submit_rec(campo) 
   { 
      if (nm_gp_ini == "ini" && (campo == "ini" || campo == nm_gp_rec_ini)) 
      { 
          return; 
      } 
      if (nm_gp_fim == "fim" && (campo == "fim" || campo == nm_gp_rec_fim)) 
      { 
          return; 
      } 
      nm_gp_submit_ajax("rec", campo); 
   } 
   function nm_gp_submit_qsearch(pos) 
   { 
      var out_qsearch = "";
       out_qsearch = document.getElementById('fast_search_f0_' + pos).value;
       out_qsearch += "_SCQS_" + document.getElementById('cond_fast_search_f0_' + pos).value;
       out_qsearch += "_SCQS_" + document.getElementById('SC_fast_search_' + pos).value;
       ajax_navigate('fast_search', out_qsearch); 
   } 
   function nm_gp_submit_ajax(opc, parm) 
   { 
      ajax_navigate(opc, parm); 
   } 
   function nm_gp_submit2(campo) 
   { 
      nm_gp_submit_ajax("ordem", campo); 
   } 
   function nm_gp_submit3(parms, parm_acum, opc, ancor) 
   { 
      document.F3.target               = "_self"; 
      document.F3.nmgp_parms.value     = parms ;
      document.F3.nmgp_parm_acum.value = parm_acum ;
      document.F3.nmgp_opcao.value     = opc ;
      document.F3.nmgp_url_saida.value = "";
      document.F3.action               = "ConPublicacaoGeral.php"  ;
      if (ancor != null) {
         ajax_save_ancor("F3", ancor);
      } else {
          document.F3.submit() ;
      } 
   } 
   function nm_gp_submit4(apl_lig, apl_saida, parms, target, opc, apl_name, ancor) 
   { 
      document.F3.target = target; 
      document.F3.action = apl_lig  ;
      if (opc == 'igual' || opc == 'novo') 
      {
          document.F3.nmgp_opcao.value = opc;
      }
      else
      if (opc != null && opc != '') 
      {
          document.F3.nmgp_opcao.value = "grid" ;
      }
      else
      {
          document.F3.nmgp_opcao.value = "igual" ;
      }
      document.F3.nmgp_url_saida.value   = apl_saida ;
      document.F3.nmgp_parms.value       = parms ;
      if (target == '_blank') 
      {
          NM_ancor_ult_lig = ancor;
          document.F3.nmgp_outra_jan.value = "true" ;
          window.open('','jan_sc','location=no,menubar=no,resizable,scrollbars,status=no,toolbar=no');
          document.F3.target = "jan_sc"; 
      }
      if (ancor != null && target == '_self') {
         ajax_save_ancor("F3", ancor);
      } else {
          document.F3.submit() ;
      } 
   } 
   function nm_gp_submit5(apl_lig, apl_saida, parms, target, opc, modal_h, modal_w, m_confirm, apl_name, ancor) 
   { 
      parms = parms.replace(/@percent@/g, "%"); 
      if (m_confirm != null && m_confirm != '') 
      { 
          if (confirm(m_confirm))
          { }
          else
          {
             return;
          }
      }
      if (apl_lig.substr(0, 7) == "http://" || apl_lig.substr(0, 8) == "https://")
      {
          if (target == '_blank') 
          {
              window.open (apl_lig);
          }
          else
          {
              window.location = apl_lig;
          }
          return;
      }
      if (target == 'modal' || target == 'modal_rpdf') 
      {
          NM_ancor_ult_lig = ancor;
          par_modal = '?script_case_init=7840&script_case_session=nuvh154aavroihs40r2786vf75&nmgp_outra_jan=true&nmgp_url_saida=modal&SC_lig_apl_orig=ConPublicacaoGeral';
          if (opc != null && opc != '') 
          {
              par_modal += '&nmgp_opcao=grid';
          }
          if (parms != null && parms != '') 
          {
              par_modal += '&nmgp_parms=' + parms;
          }
          if (target == 'modal') 
          {
               parent.tb_show('', apl_lig + par_modal + '&TB_iframe=true&modal=true&height=' + modal_h + '&width=' + modal_w, '');
          }
          else 
          {
               parent.tb_show('', apl_lig + par_modal + '&TB_iframe=true&height=' + modal_h + '&width=' + modal_w, '');
          }
          return;
      }
      document.F3.target = target; 
      if (target == '_blank') 
      {
          NM_ancor_ult_lig = ancor;
          document.F3.nmgp_outra_jan.value = "true" ;
          window.open('','jan_sc','location=no,menubar=no,resizable,scrollbars,status=no,toolbar=no');
          document.F3.target = "jan_sc"; 
      }
      document.F3.action = apl_lig  ;
      if (opc != null && opc != '') 
      {
          document.F3.nmgp_opcao.value = "grid" ;
      }
      else
      {
          document.F3.nmgp_opcao.value = "" ;
      }
      document.F3.nmgp_url_saida.value = apl_saida ;
      document.F3.nmgp_parms.value     = parms ;
      if (ancor != null && target == '_self') {
         ajax_save_ancor("F3", ancor);
      } else {
          document.F3.submit() ;
      } 
      document.F3.nmgp_outra_jan.value   = "" ;
   } 
   function nm_gp_submit6(apl_lig, apl_saida, parms, target, pos, alt, larg, opc, modal_h, modal_w, m_confirm, apl_name, ancor) 
   { 
      if (apl_lig.substr(0, 7) == "http://" || apl_lig.substr(0, 8) == "https://")
      {
          if (target == '_blank') 
          {
              window.open (apl_lig);
          }
          else
          {
              window.location = apl_lig;
          }
          return;
      }
      if (pos == "A") {obj = document.getElementById('nmsc_iframe_liga_A_ConPublicacaoGeral');} 
      if (pos == "B") {obj = document.getElementById('nmsc_iframe_liga_B_ConPublicacaoGeral');} 
      if (pos == "E") {obj = document.getElementById('nmsc_iframe_liga_E_ConPublicacaoGeral');} 
      if (pos == "D") {obj = document.getElementById('nmsc_iframe_liga_D_ConPublicacaoGeral');} 
      obj.style.height = (alt == parseInt(alt)) ? alt + 'px' : alt;
      obj.style.width  = (larg == parseInt(larg)) ? larg + 'px' : larg;
      document.F3.target = target; 
      document.F3.action = apl_lig  ;
      if (opc != null && opc != '') 
      {
          document.F3.nmgp_opcao.value = "grid" ;
      }
      else
      {
          document.F3.nmgp_opcao.value = "" ;
      }
      document.F3.nmgp_url_saida.value = apl_saida ;
      document.F3.nmgp_parms.value     = parms ;
      if (ancor != null && target == '_self') {
         ajax_save_ancor("F3", ancor);
      } else {
          document.F3.submit() ;
      } 
   } 
   function nm_submit_modal(parms, t_parent) 
   { 
      if (t_parent == 'S' && typeof parent.tb_show == 'function')
      { 
           parent.tb_show('', parms, '');
      } 
      else
      { 
         tb_show('', parms, '');
      } 
   } 
   function nm_move(tipo) 
   { 
      document.F6.target = "_self"; 
      document.F6.submit() ;
      return;
   } 
   function nm_gp_move(x, y, z, p, g) 
   { 
       document.F3.action           = "ConPublicacaoGeral.php"  ;
       document.F3.nmgp_parms.value = "SC_null" ;
       document.F3.nmgp_orig_pesq.value = "" ;
       document.F3.nmgp_url_saida.value = "" ;
       document.F3.nmgp_opcao.value = x; 
       document.F3.nmgp_outra_jan.value = "" ;
       document.F3.target = "_self"; 
       if (y == 1) 
       {
           document.F3.target = "_blank"; 
       }
       if ("busca" == x)
       {
           document.F3.nmgp_orig_pesq.value = z; 
           z = '';
       }
       if (z != null && z != '') 
       { 
           document.F3.nmgp_tipo_pdf.value = z; 
       } 
       if ("xls" == x)
       {
       }
       if ("pdf" == x)
       {
           window.location = "/consulta/ConPublicacaoGeral/ConPublicacaoGeral_iframe.php?nmgp_parms=@SC_par@7840@SC_par@ConPublicacaoGeral@SC_par@379d11ed89d8caba816339b02ef3e2ff&sc_tp_pdf=" + z + "&sc_parms_pdf=" + p + "&sc_graf_pdf=" + g;
       }
       else
       {
           if ((x == 'igual' || x == 'edit') && NM_ancor_ult_lig != "")
           {
                ajax_save_ancor("F3", NM_ancor_ult_lig);
                NM_ancor_ult_lig = "";
            } else {
                document.F3.submit() ;
            } 
       }
   } 
   function nm_gp_print_conf(tp, cor)
   {
       window.open('/consulta/ConPublicacaoGeral/ConPublicacaoGeral_iframe_prt.php?path_botoes=/consulta/_lib/img&script_case_init=7840&script_case_session=nuvh154aavroihs40r2786vf75&opcao=print&tp_print=' + tp + '&cor_print=' + cor,'','location=no,menubar,resizable,scrollbars,status=no,toolbar');
   }
   nm_img = new Image();
   function nm_mostra_img(imagem, altura, largura)
   {
       tb_show("", imagem, "");
   }
   function nm_mostra_doc(campo1, campo2, campo3, campo4)
   {
       while (campo2.lastIndexOf("&") != -1)
       {
          campo2 = campo2.replace("&" , "**Ecom**");
       }
       while (campo2.lastIndexOf("#") != -1)
       {
          campo2 = campo2.replace("#" , "**Jvel**");
       }
       while (campo2.lastIndexOf("+") != -1)
       {
          campo2 = campo2.replace("+" , "**Plus**");
       }
       NovaJanela = window.open (campo4 + "?script_case_init=7840&script_case_session=nuvh154aavroihs40r2786vf75&nm_cod_doc=" + campo1 + "&nm_nome_doc=" + campo2 + "&nm_cod_apl=" + campo3, "ScriptCase", "resizable");
   }
   function nm_escreve_window()
   {
      document.F5.action = "ConPublicacaoGeral_fim.php";
      document.F5.submit();
   }
   function nm_open_popup(parms)
   {
       NovaJanela = window.open (parms, '', 'resizable, scrollbars');
   }
   </script>
   <TABLE id="main_table_grid" cellspacing=0 cellpadding=0 align="center" valign="top"  width="100%">
     <TR>
       <TD>
       <div class="scGridBorder">
  <div id="id_div_process" style="display: none; margin: 10px; whitespace: nowrap" class="scFormProcessFixed"><span class="scFormProcess"><img border="0" src="/consulta/_lib/img/scriptcase__NM__ajax_load.gif" align="absmiddle" />&nbsp;Processando Aguarde...</span></div>
  <div id="id_div_process_block" style="display: none; margin: 10px; whitespace: nowrap"><span class="scFormProcess"><img border="0" src="/consulta/_lib/img/scriptcase__NM__ajax_load.gif" align="absmiddle" />&nbsp;Processando Aguarde...</span></div>
  <div id="id_fatal_error" class="scGridLabel" style="display: none; position: absolute"></div>
       <TABLE width='100%' cellspacing=0 cellpadding=0>
    <TR>
    <TD  colspan=3 style="padding: 0px; border-width: 0px; vertical-align: top;">
     <iframe class="css_iframes" id="nmsc_iframe_liga_A_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_A_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
    </TD>
    </TR>
    <TR>
    <TD style="padding: 0px; border-width: 0px; vertical-align: top;">
     <iframe class="css_iframes" id="nmsc_iframe_liga_E_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_E_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
    </TD>
    <td style="padding: 0px;  vertical-align: top;"><table style="padding: 0px; border-spacing: 0px; border-width: 0px; vertical-align: top;" width="100%"><tr>
    <TD colspan=3 style="padding: 0px; border-width: 0px; vertical-align: top;" width=1>
     <iframe class="css_iframes" id="nmsc_iframe_liga_AL_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_AL_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
    </TD>
    </TR>
 <TR>
  <TD id='sc_grid_content'  colspan=3>
    <table width='100%' cellspacing=0 cellpadding=0>
      <tr style="display: none">
      <td>
      <form id="id_F0_top" name="F0_top" method="post" action="ConPublicacaoGeral.php" target="_self"> 
      <input type="text" id="id_sc_truta_f0_top" name="sc_truta_f0_top" value=""/> 
      <input type="hidden" id="script_init_f0_top" name="script_case_init" value="7840"/> 
      <input type=hidden id="script_session_f0_top" name="script_case_session" value="nuvh154aavroihs40r2786vf75"/>
      <input type="hidden" id="opcao_f0_top" name="nmgp_opcao" value="muda_qt_linhas"/> 
      </td></tr><tr>
       <td id="sc_grid_toobar_top"  colspan=3 class="scGridTabelaTd" valign="top"> 
        <table class="scGridToolbar" style="padding: 0px; border-spacing: 0px; border-width: 0px; vertical-align: top;" width="100%" valign="top">
         <tr> 
          <td class="scGridToolbarPadding" nowrap valign="middle" align="left" width="33%"> 
         </td> 
          <td class="scGridToolbarPadding" nowrap valign="middle" align="center" width="33%"> 
          <a  href="javascript: sc_btn_AgruparPDFs()" id="sc_AgruparPDFs_top" onClick="sc_btn_AgruparPDFs(); return false;" class="scButton_default" style="vertical-align: middle; display:inline-block;">AgruparPDFs</a>
 
         </td> 
          <td class="scGridToolbarPadding" nowrap valign="middle" align="right" width="33%"> 
         </td> 
        </tr> 
       </table> 
      </td> 
     </tr> 
      <tr style="display: none">
      <td> 
     </form> 
      </td> 
     </tr> 
     <tr id="sc_id_save_grid_placeholder_top" style="display: none">
      <td colspan=3>
      </td>
     </tr>
     <tr id="sc_id_groupby_placeholder_top" style="display: none">
      <td colspan=3>
      </td>
     </tr>
     <tr id="sc_id_sel_campos_placeholder_top" style="display: none">
      <td colspan=3>
      </td>
     </tr>
     <tr id="sc_id_order_campos_placeholder_top" style="display: none">
      <td colspan=3>
      </td>
     </tr>
 <TR> 
    <TD  colspan=3>
    <TABLE cellspacing=0 cellpadding=0 width='100%'>
    <TD style="padding: 0px; border-width: 0px; vertical-align: top;" width=1>
     <iframe class="css_iframes" id="nmsc_iframe_liga_EL_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_EL_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
    </TD>
    <TD style="padding: 0px; border-width: 0px; vertical-align: top;"><TABLE style="padding: 0px; border-spacing: 0px; border-width: 0px;" width="100%"><TR>
  <TD id="sc_grid_body" class="scGridTabelaTd" style="vertical-align: top;text-align: center;" width="100%">
       <div id="div_FBtn_Run" style="display: none"> 
       <form name="FBtn_Run" method="post" action="ConPublicacaoGeral.php" target="_self">
        <input type="hidden" name="nmgp_opcao" value="formphp"> 
        <input type="hidden" name="rec" value=""> 
        <input type="hidden" name="nm_call_php" value=""> 
        <input type="hidden" name="nm_run_opt_sel" value=""> 
        <input type="hidden" name="script_case_init" value="7840"> 
        <input type=hidden name="script_case_session" value="nuvh154aavroihs40r2786vf75"/>
       </div> 
   <TABLE class="scGridTabela" id="sc-ui-grid-body-6898a72d" align="center"  id="apl_ConPublicacaoGeral#?#1" width="100%">
    <TR id="tit_ConPublicacaoGeral__SCCS__1" align="center" class="scGridLabel sc-ui-grid-header-row sc-ui-grid-header-row-ConPublicacaoGeral-1">
     <TD class="scGridLabelFont"  style="" ><input type=checkbox id="NM_ck_run0" name="NM_ck_grid[]" value="0" style="display:''" onClick="nm_marca_check_grid(this)"></TD>
     <TD class="scGridLabelFont css_numedicao_label"  style="" >N° Edição</TD>
     <TD class="scGridLabelFont css_ano_label"  style="" >Ano</TD>
     <TD class="scGridLabelFont css_data_label"  style="" >Data</TD>
     <TD class="scGridLabelFont css_nomemunicipio_label"  style="" >Município</TD>
     <TD class="scGridLabelFont css_nomeentidade_label"  style="" >Entidade</TD>
     <TD class="scGridLabelFont css_nomecategoria_label"  style="" >Categoria</TD>
     <TD class="scGridLabelFont css_nomedoc_label"  style="" >Documento</TD>
     <TD class="scGridLabelFont css_nomearq_label"  style="" >Arquivo</TD>
     <TD class="scGridLabelFont css_codigo_label"  style="" >Identificador</TD>
</TR>
    <TR  class="scGridFieldOdd" onmouseover="over_tr(this, 'scGridFieldOdd');" onmouseout="out_tr(this, 'scGridFieldOdd');" onclick="click_tr(this, 'scGridFieldOdd');" id="SC_ancor1">
     <TD rowspan="1" class="scGridFieldOddFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run1" class="sc-ui-check-run" name="NM_ck_grid[]" value="1" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5230/DM_5230.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_numedicao_grid_line" style="">5230</a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_1">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_1">02/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_1">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_1">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_1">Encerramento</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_1">Encerramento do Prazo Para Registro de Chapas</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5230/DM_5230_091_Campo_Maior_Encerramento_do_Prazo_Para_Registro_de_Chapas_pag_469.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_1">07384CCD29137A16</span></TD>
</TR>
    <TR  class="scGridFieldEven" onmouseover="over_tr(this, 'scGridFieldEven');" onmouseout="out_tr(this, 'scGridFieldEven');" onclick="click_tr(this, 'scGridFieldEven');" id="SC_ancor2">
     <TD rowspan="1" class="scGridFieldEvenFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run2" class="sc-ui-check-run" name="NM_ck_grid[]" value="2" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_2">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_2">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_2">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_2">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_2">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_2">Portaria 001-25</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_193_Campo_Maior_Portaria_001-25_pag_265.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_2">167C49564FDD3AE5</span></TD>
</TR>
    <TR  class="scGridFieldOdd" onmouseover="over_tr(this, 'scGridFieldOdd');" onmouseout="out_tr(this, 'scGridFieldOdd');" onclick="click_tr(this, 'scGridFieldOdd');" id="SC_ancor3">
     <TD rowspan="1" class="scGridFieldOddFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run3" class="sc-ui-check-run" name="NM_ck_grid[]" value="3" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_3">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_3">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_3">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_3">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_3">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_3">Portaria 002-25</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_194_Campo_Maior_Portaria_002-25_pag_265.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_3">167C49564FDD3AED</span></TD>
</TR>
    <TR  class="scGridFieldEven" onmouseover="over_tr(this, 'scGridFieldEven');" onmouseout="out_tr(this, 'scGridFieldEven');" onclick="click_tr(this, 'scGridFieldEven');" id="SC_ancor4">
     <TD rowspan="1" class="scGridFieldEvenFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run4" class="sc-ui-check-run" name="NM_ck_grid[]" value="4" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_4">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_4">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_4">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_4">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_4">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_4">Portaria 003-25</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_195_Campo_Maior_Portaria_003-25_pag_266.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_4">030E7CC132513AF2</span></TD>
</TR>
    <TR  class="scGridFieldOdd" onmouseover="over_tr(this, 'scGridFieldOdd');" onmouseout="out_tr(this, 'scGridFieldOdd');" onclick="click_tr(this, 'scGridFieldOdd');" id="SC_ancor5">
     <TD rowspan="1" class="scGridFieldOddFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run5" class="sc-ui-check-run" name="NM_ck_grid[]" value="5" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_5">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_5">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_5">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_5">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_5">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_5">Portaria 004-25</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_196_Campo_Maior_Portaria_004-25_pag_266.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_5">13B5BE6594C93AF7</span></TD>
</TR>
    <TR  class="scGridFieldEven" onmouseover="over_tr(this, 'scGridFieldEven');" onmouseout="out_tr(this, 'scGridFieldEven');" onclick="click_tr(this, 'scGridFieldEven');" id="SC_ancor6">
     <TD rowspan="1" class="scGridFieldEvenFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run6" class="sc-ui-check-run" name="NM_ck_grid[]" value="6" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_6">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_6">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_6">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_6">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_6">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_6">Portaria 005-25</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_197_Campo_Maior_Portaria_005-25_pag_266.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_6">167C49564FDD3AFD</span></TD>
</TR>
    <TR  class="scGridFieldOdd" onmouseover="over_tr(this, 'scGridFieldOdd');" onmouseout="out_tr(this, 'scGridFieldOdd');" onclick="click_tr(this, 'scGridFieldOdd');" id="SC_ancor7">
     <TD rowspan="1" class="scGridFieldOddFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run7" class="sc-ui-check-run" name="NM_ck_grid[]" value="7" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_7">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_7">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_7">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_7">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_7">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_7">Portaria 006-24</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_198_Campo_Maior_Portaria_006-24_pag_266.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_7">10EF3374D9B53B01</span></TD>
</TR>
    <TR  class="scGridFieldEven" onmouseover="over_tr(this, 'scGridFieldEven');" onmouseout="out_tr(this, 'scGridFieldEven');" onclick="click_tr(this, 'scGridFieldEven');" id="SC_ancor8">
     <TD rowspan="1" class="scGridFieldEvenFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run8" class="sc-ui-check-run" name="NM_ck_grid[]" value="8" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_8">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_8">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_8">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_8">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_8">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_8">Portaria 007-25</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_199_Campo_Maior_Portaria_007-25_pag_267.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_8">089B92A2A8793B0F</span></TD>
</TR>
    <TR  class="scGridFieldOdd" onmouseover="over_tr(this, 'scGridFieldOdd');" onmouseout="out_tr(this, 'scGridFieldOdd');" onclick="click_tr(this, 'scGridFieldOdd');" id="SC_ancor9">
     <TD rowspan="1" class="scGridFieldOddFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run9" class="sc-ui-check-run" name="NM_ck_grid[]" value="9" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_9">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_9">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_9">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_9">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_9">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_9">Portaria 008-25</span></TD>
     <TD rowspan="1" class="scGridFieldOddFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_200_Campo_Maior_Portaria_008-25_pag_267.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldOddLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldOddFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_9">0CC5630BC1173B14</span></TD>
</TR>
    <TR  class="scGridFieldEven" onmouseover="over_tr(this, 'scGridFieldEven');" onmouseout="out_tr(this, 'scGridFieldEven');" onclick="click_tr(this, 'scGridFieldEven');" id="SC_ancor10">
     <TD rowspan="1" class="scGridFieldEvenFont"  style="" NOWRAP align="left" valign="top" WIDTH="1px"  HEIGHT="0px"> <input type="checkbox" id="NM_ck_run10" class="sc-ui-check-run" name="NM_ck_grid[]" value="10" style="align:left;vertical-align:middle;font-weight:bold;" /></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_numedicao_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232.pdf', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_numedicao_grid_line" style="">5232</a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_ano_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_ano_10">2.025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_data_grid_line"  style="" NOWRAP align="" valign=""   HEIGHT="0px"><span id="id_sc_field_data_10">06/01/2025</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomemunicipio_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomemunicipio_10">Campo Maior</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomeentidade_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomeentidade_10">Camara</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomecategoria_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomecategoria_10">Portaria</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomedoc_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_nomedoc_10">Portaria 009-25</span></TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_nomearq_grid_line"  style=""  align="" valign=""   HEIGHT="0px">
<a href="javascript:nm_gp_submit5('http://www.diarioficialdosmunicipios.org/intranet/_lib/file/doc/pdfs/novo/5232/DM_5232_201_Campo_Maior_Portaria_009-25_pag_267.pdf ', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php', 'OrScLink?#?1?@?Baixar?#??@?', '_blank', '', '440', '630')" onMouseover="nm_mostra_hint(this, event, '')" onMouseOut="nm_apaga_hint()" class="scGridFieldEvenLink css_nomearq_grid_line" style=""><img border="0" src="/consulta/_lib/img/scriptcase__NM__ico__NM__nm_icon_pdf.gif"/></a>
</TD>
     <TD rowspan="1" class="scGridFieldEvenFont css_codigo_grid_line"  style=""  align="" valign=""   HEIGHT="0px"><span id="id_sc_field_codigo_10">05D507B1ED653B19</span></TD>
</TR>
</TABLE>       </form>
</TD></tr></TABLE></TD>
<TD style="padding: 0px; border-width: 0px;" valign="top" width=1>
     <iframe class="css_iframes" id="nmsc_iframe_liga_DL_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_DL_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
</TD>
    <TD style="padding: 0px; border-width: 0px; vertical-align: top;">
     <iframe class="css_iframes" id="nmsc_iframe_liga_D_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_D_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
    </TD>
    </TR>
    </TABLE>
    </TD>
    </TR>
     <tr id="sc_id_save_grid_placeholder_bot" style="display: none">
      <td colspan=3>
      </td>
     </tr>
     <tr id="sc_id_groupby_placeholder_bot" style="display: none">
      <td colspan=3>
      </td>
     </tr>
     <tr id="sc_id_sel_campos_placeholder_bot" style="display: none">
      <td colspan=3>
      </td>
     </tr>
     <tr id="sc_id_order_campos_placeholder_bot" style="display: none">
      <td colspan=3>
      </td>
     </tr>
      <tr style="display: none">
      <td>
      <form id="id_F0_bot" name="F0_bot" method="post" action="ConPublicacaoGeral.php" target="_self"> 
      <input type="text" id="id_sc_truta_f0_bot" name="sc_truta_f0_bot" value=""/> 
      <input type="hidden" id="script_init_f0_bot" name="script_case_init" value="7840"/> 
      <input type=hidden id="script_session_f0_bot" name="script_case_session" value="nuvh154aavroihs40r2786vf75"/>
      <input type="hidden" id="opcao_f0_bot" name="nmgp_opcao" value="muda_qt_linhas"/> 
      </td></tr><tr>
       <td id="sc_grid_toobar_bot"  colspan=3 class="scGridTabelaTd" valign="top"> 
        <table class="scGridToolbar" style="padding: 0px; border-spacing: 0px; border-width: 0px; vertical-align: top;" width="100%" valign="top">
         <tr> 
          <td class="scGridToolbarPadding" nowrap valign="middle" align="left" width="33%"> 
           <a  id="first_bot" border="0px" title="Retornar ao início" style="vertical-align: middle; display:inline-block;" align="absmiddle" onClick="nm_gp_submit_rec('ini'); return false;"><img  id="id_img_first_bot" src="/consulta/_lib/img/scriptcase__NM__nm_Scriptcase5_Silver_bcons_inicio_off.gif" style="border-width: 0; cursor: pointer" /></a>
 
           <a  id="back_bot" border="0px" title="Retornar um registro" style="vertical-align: middle; display:inline-block;" align="absmiddle" onClick="nm_gp_submit_rec('0'); return false;"><img  id="id_img_back_bot" src="/consulta/_lib/img/scriptcase__NM__nm_Scriptcase5_Silver_bcons_retorna_off.gif" style="border-width: 0; cursor: pointer" /></a>
 
           <a  id="forward_bot" border="0px" title="Avançar para o próximo registro" style="vertical-align: middle; display:inline-block;" align="absmiddle" onClick="nm_gp_submit_rec('11'); return false;"><img  id="id_img_forward_bot" src="/consulta/_lib/img/scriptcase__NM__nm_Scriptcase5_Silver_bcons_avanca.gif" style="border-width: 0; cursor: pointer" /></a>
 
           <a  id="last_bot" border="0px" title="Avançar para o final" style="vertical-align: middle; display:inline-block;" align="absmiddle" onClick="nm_gp_submit_rec('fim'); return false;"><img  id="id_img_last_bot" src="/consulta/_lib/img/scriptcase__NM__nm_Scriptcase5_Silver_bcons_final.gif" style="border-width: 0; cursor: pointer" /></a>
 
         </td> 
          <td class="scGridToolbarPadding" nowrap valign="middle" align="center" width="33%"> 
           <a  href="javascript: nm_gp_move('xls', '0')" id="xls_bot" onClick="nm_gp_move('xls', '0'); return false;" class="scButton_default" title="Gerar XLS (Excel, BrOffice)" style="vertical-align: middle; display:inline-block;">XLS</a>
 
           <a  id="print_bot" onClick="tb_show('', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral_config_print.php?nm_opc=AM&nm_cor=AM&language=pt_br&nm_page=7840&KeepThis=true&TB_iframe=true&modal=true')"  class="thickbox scButton_default" title="Imprimir" style="vertical-align: middle; display:inline-block;">Imprimir</a>
 
           <a  id="pdf_bot" onClick="tb_show('', '/consulta/ConPublicacaoGeral/ConPublicacaoGeral_config_pdf.php?nm_opc=pdf&nm_target=0&nm_cor=cor&papel=8&lpapel=0&apapel=0&orientacao=2&bookmarks=XX&largura=1200&conf_larg=S&conf_fonte=10&grafico=XX&language=pt_br&conf_socor=S&KeepThis=true&TB_iframe=true&modal=true')"  class="thickbox scButton_default" title="Gerar PDF" style="vertical-align: middle; display:inline-block;">Gerar PDF</a>
 
           <span class="css_toolbar_obj" style="border:0px;">[1 a 10 de 270]</span>
         </td> 
          <td class="scGridToolbarPadding" nowrap valign="middle" align="right" width="33%"> 
           <a  href="javascript: var rec_nav = ((document.getElementById('rec_f0_bot').value - 1) * 10) + 1; nm_gp_submit_ajax('muda_rec_linhas', rec_nav)" id="brec_bot" onClick="var rec_nav = ((document.getElementById('rec_f0_bot').value - 1) * 10) + 1; nm_gp_submit_ajax('muda_rec_linhas', rec_nav); return false;" class="scButton_default" title="Ir para a linha" style="vertical-align: middle; display:inline-block;">Ir para</a>
 
          <input id="rec_f0_bot" type="text" class="css_toolbar_obj" name="rec" value="1" style="width:25px;vertical-align: middle;"/> 
           <a  href="javascript: nm_gp_submit_ajax('muda_qt_linhas', document.getElementById('quant_linhas_f0_bot').value)" id="qtlin_bot" onClick="nm_gp_submit_ajax('muda_qt_linhas', document.getElementById('quant_linhas_f0_bot').value); return false;" class="scButton_default" title="Alterar quantidade de linhas da Grid" style="vertical-align: middle; display:inline-block;">Visualizar</a>
 
          <input type="text" class="css_toolbar_obj" id="quant_linhas_f0_bot" name="nmgp_quant_linhas" value="10" style="width:25px;vertical-align: middle;"/> 
         </td> 
        </tr> 
       </table> 
      </td> 
     </tr> 
      <tr style="display: none">
      <td> 
     </form> 
      </td> 
     </tr> 
   </table>
  </TD>
 </TR>
     <tr><td colspan=3  class="scGridTabelaTd" style="vertical-align: top"> 
     <iframe class="css_iframes" id="nmsc_iframe_liga_B_ConPublicacaoGeral" marginWidth="0px" marginHeight="0px" frameborder="0" valign="top" height="0px" width="0px" name="nm_iframe_liga_B_ConPublicacaoGeral" scrolling="auto" src="NM_Blank_Page.htm"></iframe>
     </td></tr> 
   </TABLE>
   </div>
   </TR>
   </TD>
   </TABLE>
   <div id="sc-id-fixedheaders-placeholder" style="display: none; position: fixed; top: 0"></div>
   </body>
   <script type="text/javascript">
   NM_ancor_ult_lig = '';
   function NM_liga_tbody(tbody, Obj, Apl)
   {
      Nivel = parseInt (Obj[tbody].substr(3));
      for (ind = tbody + 1; ind < Obj.length; ind++)
      {
           Nv = parseInt (Obj[ind].substr(3));
           Tp = Obj[ind].substr(0, 3);
           if (Nivel == Nv && Tp == 'top')
           {
               break;
           }
           if (((Nivel + 1) == Nv && Tp == 'top') || (Nivel == Nv && Tp == 'bot'))
           {
               document.getElementById('tbody_' + Apl + '_' + ind + '_' + Tp).style.display='';
           } 
      }
   }
   function NM_apaga_tbody(tbody, Obj, Apl)
   {
      Nivel = Obj[tbody].substr(3);
      for (ind = tbody + 1; ind < Obj.length; ind++)
      {
           Nv = Obj[ind].substr(3);
           Tp = Obj[ind].substr(0, 3);
           if ((Nivel == Nv && Tp == 'top') || Nv < Nivel)
           {
               break;
           }
           if ((Nivel != Nv) || (Nivel == Nv && Tp == 'bot'))
           {
               document.getElementById('tbody_' + Apl + '_' + ind + '_' + Tp).style.display='none';
               if (Tp == 'top')
               {
                   document.getElementById('b_open_' + Apl + '_' + ind).style.display='';
                   document.getElementById('b_close_' + Apl + '_' + ind).style.display='none';
               } 
           } 
      }
   }
   NM_obj_ant = '';
   function NM_apaga_div_lig(obj_nome)
   {
      if (NM_obj_ant != '')
      {
          NM_obj_ant.style.display='none';
      }
      obj = document.getElementById(obj_nome);
      NM_obj_ant = obj;
      ind_time = setTimeout("obj.style.display='none'", 300);
      return ind_time;
   }
  $(window).scroll(function() {
   scSetFixedHeaders();
  }).resize(function() {
   scSetFixedHeaders();
  });
   nm_gp_fim = "";
   </script>
   </HTML>
