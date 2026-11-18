using System.Text.Json;
using System.Text.RegularExpressions;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;
using DocumentFormat.OpenXml.Wordprocessing;
using A = DocumentFormat.OpenXml.Drawing;
using DW = DocumentFormat.OpenXml.Drawing.Wordprocessing;
using PIC = DocumentFormat.OpenXml.Drawing.Pictures;

class Entry {
    public string Title { get; set; } = "";
    public string Bookmark { get; set; } = "";
    public int Level { get; set; }
}

class Program {
    static MainDocumentPart main = null!;
    static Body body = null!;
    static Dictionary<string,string> pages = new();
    static readonly List<Entry> entries = new();
    static int bookmarkId = 1;
    static JsonElement rootConfig;
    static string ConfigText(string key, string fallback) => rootConfig.TryGetProperty(key, out var value) ? value.GetString()! : fallback;
    const int PageWidth = 12240, PageHeight = 15840, ContentWidth = 8640;
    static readonly string Font = "Times New Roman";

    static RunFonts Fonts() => new() { Ascii=Font, HighAnsi=Font, EastAsia=Font, ComplexScript=Font };
    static RunProperties RPr(int size=24, bool bold=false) {
        var r = new RunProperties();
        r.Append(Fonts());
        if (bold) r.Append(new Bold(), new BoldComplexScript());
        r.Append(new Color{Val="000000"},new FontSize{Val=size.ToString()},
                 new FontSizeComplexScript{Val=size.ToString()},new Languages{Val="vi-VN"});
        return r;
    }
    static Run Run(string text, int size=24, bool bold=false) => new(RPr(size,bold),new Text(text){Space=SpaceProcessingModeValues.Preserve});
    static Paragraph P(string text, string style="Normal", JustificationValues? align=null, bool bold=false, int size=24) {
        var pr=new ParagraphProperties(new ParagraphStyleId{Val=style});
        if (align.HasValue) pr.Append(new Justification{Val=align.Value});
        return new Paragraph(pr, Run(text,size,bold));
    }
    static void AddP(string text) => body.Append(P(text));
    static Style Style(string id,string name,int size,bool bold,int line,int before,int after,JustificationValues align,int? outline=null,int firstLine=0) {
        var pr=new StyleParagraphProperties();
        if(outline.HasValue) pr.Append(new KeepNext(),new KeepLines());
        pr.Append(new WidowControl(),new SpacingBetweenLines{Before=before.ToString(),After=after.ToString(),Line=line.ToString(),LineRule=LineSpacingRuleValues.Auto},
                  new Indentation{FirstLine=firstLine.ToString()},new Justification{Val=align});
        if(outline.HasValue) pr.Append(new OutlineLevel{Val=outline.Value});
        var rp=new StyleRunProperties(); rp.Append(Fonts());
        if(bold) rp.Append(new Bold(),new BoldComplexScript());
        rp.Append(new Color{Val="000000"},new FontSize{Val=size.ToString()},new FontSizeComplexScript{Val=size.ToString()},new Languages{Val="vi-VN"});
        var s=new Style(new StyleName{Val=name});
        if(id!="Normal") s.Append(new BasedOn{Val="Normal"},new NextParagraphStyle{Val="Normal"});
        s.Append(new PrimaryStyle(),pr,rp); s.Type=StyleValues.Paragraph;s.StyleId=id;
        if(id=="Normal") s.Default=true;
        return s;
    }
    static void Styles() {
        var sp=main.AddNewPart<StyleDefinitionsPart>(); sp.Styles=new Styles();
        sp.Styles.Append(new DocDefaults(new RunPropertiesDefault(new RunPropertiesBaseStyle(Fonts(),new Color{Val="000000"},new FontSize{Val="24"},new FontSizeComplexScript{Val="24"},new Languages{Val="vi-VN"})),
                         new ParagraphPropertiesDefault(new ParagraphPropertiesBaseStyle(new SpacingBetweenLines{Line="360",LineRule=LineSpacingRuleValues.Auto,After="0"},new Indentation{FirstLine="720"}))));
        sp.Styles.Append(Style("Normal","Normal",24,false,360,0,0,JustificationValues.Both,null,720),
          Style("Title","Title",36,true,300,240,180,JustificationValues.Center),
          Style("Heading1","heading 1",24,true,360,240,180,JustificationValues.Center,0),
          Style("Heading2","heading 2",24,true,360,180,60,JustificationValues.Left,1),
          Style("Heading3","heading 3",24,true,360,120,60,JustificationValues.Left,2),
          Style("Cover","Cover",24,false,360,0,80,JustificationValues.Center),
          Style("TableText","Table Text",23,false,264,0,40,JustificationValues.Left),
          Style("Caption","Caption",24,true,300,180,80,JustificationValues.Center),
          Style("TOC1","toc 1",22,true,240,60,0,JustificationValues.Left),
          Style("TOC2","toc 2",22,false,240,0,0,JustificationValues.Left),
          Style("Bibliography","Bibliography",24,false,360,0,160,JustificationValues.Left),
          Style("Header","Header",20,false,240,0,0,JustificationValues.Right),
          Style("Footer","Footer",22,false,240,0,0,JustificationValues.Center));
        sp.Styles.Append(Style("Code","Code",22,false,240,0,0,JustificationValues.Left));
        sp.Styles.Append(new Style(new StyleName{Val="Hyperlink"},new StyleRunProperties(Fonts(),new Color{Val="000000"})){Type=StyleValues.Character,StyleId="Hyperlink"});
    }
    static void Numbering() {
        var np=main.AddNewPart<NumberingDefinitionsPart>();np.Numbering=new Numbering();
        var abs=new AbstractNum{AbstractNumberId=1};
        abs.Append(new MultiLevelType{Val=MultiLevelValues.Multilevel});
        for(int i=0;i<3;i++) {
            var level=new Level{LevelIndex=i};
            level.Append(new StartNumberingValue{Val=1},new NumberingFormat{Val=NumberFormatValues.Decimal});
            if(i>0) level.Append(new LevelRestart{Val=i});
            level.Append(new ParagraphStyleIdInLevel{Val=$"Heading{i+1}"},new LevelSuffix{Val=LevelSuffixValues.Space},
                         new LevelText{Val=i==0?"CHƯƠNG %1":i==1?"%1.%2":"%1.%2.%3"},new LevelJustification{Val=LevelJustificationValues.Left});
            abs.Append(level);
        }
        np.Numbering.Append(abs,new NumberingInstance(new AbstractNumId{Val=1}){NumberID=1});
    }
    static string Footer(bool numbered) {
        var fp=main.AddNewPart<FooterPart>();var p=P("","Footer");
        if(numbered) {
            p.Append(new Run(RPr(22),new FieldChar{FieldCharType=FieldCharValues.Begin}),
                     new Run(RPr(22),new FieldCode(" PAGE "){Space=SpaceProcessingModeValues.Preserve}),
                     new Run(RPr(22),new FieldChar{FieldCharType=FieldCharValues.Separate}),Run("1",22),
                     new Run(RPr(22),new FieldChar{FieldCharType=FieldCharValues.End}));
        }
        fp.Footer=new Footer(p);fp.Footer.Save();return main.GetIdOfPart(fp);
    }
    static string Header(string text) {
        var hp=main.AddNewPart<HeaderPart>();hp.Header=new Header(P(text,"Header",size:20));hp.Header.Save();return main.GetIdOfPart(hp);
    }
    static SectionProperties Section(string header,string footer,NumberFormatValues format=default,bool restart=false) {
        var sec=new SectionProperties();
        sec.Append(new HeaderReference{Type=HeaderFooterValues.Default,Id=header},new FooterReference{Type=HeaderFooterValues.Default,Id=footer},
                   new SectionType{Val=SectionMarkValues.NextPage},new PageSize{Width=PageWidth,Height=PageHeight},
                   new PageMargin{Top=1440,Right=1440,Bottom=1440,Left=2160,Header=600,Footer=600,Gutter=0});
        if(restart) sec.Append(new PageNumberType{Format=format,Start=1});
        return sec;
    }
    static void EndSection(SectionProperties sec) {
        body.Append(new Paragraph(new ParagraphProperties(new SpacingBetweenLines{After="0",Line="20",LineRule=LineSpacingRuleValues.Exact},new SectionProperties(sec.ChildElements.Select(x=>x.CloneNode(true))))));
    }
    static void CoverP(string text,int size=24,bool bold=false,int before=0,int after=80,string style="Cover") {
        var p=P(text,style,JustificationValues.Center,bold,size);
        p.ParagraphProperties!.Append(new SpacingBetweenLines{Before=before.ToString(),After=after.ToString(),Line="360",LineRule=LineSpacingRuleValues.Auto});
        body.Append(p);
    }
    static void Cover() {
        CoverP("TRƯỜNG ........................................................",24,true);
        CoverP("KHOA .............................................................",24,true);
        CoverP(ConfigText("cover_title", "BÁO CÁO KẾ HOẠCH THỰC HIỆN ĐỀ TÀI"),32,true,660,180,"Title");
        CoverP("HỌC PHẦN TRÍ TUỆ NHÂN TẠO NÂNG CAO",24,true,0,240);
        CoverP(ConfigText("cover_topic1", "XÂY DỰNG HỆ THỐNG CỜ VUA"),30,true,0,60);
        CoverP(ConfigText("cover_topic2", "TÍCH HỢP TRÍ TUỆ NHÂN TẠO"),30,true,0,300);
        CoverP("Giảng viên hướng dẫn: .................................................",24,false,0,80);
        CoverP("Lớp: ................................    Nhóm: .............................",24,false,0,180);
        CoverP("DANH SÁCH THÀNH VIÊN THỰC HIỆN",24,true,0,100);
        Table(new[]{"STT","Họ và tên","Mã số sinh viên"},new[]{
            new[]{"1","...........................................","............................."},
            new[]{"2","...........................................","............................."},
            new[]{"3","...........................................","............................."},
            new[]{"4","...........................................","............................."}},new[]{10,53,37});
        CoverP("Thời gian thực hiện: từ ................. đến .................",24,false,180,80);
        CoverP("........................, tháng ........ năm ........",24,false,180,0);
    }
    static void Table(string[] headers,string[][] rows,int[] widths,bool keepTogether=false) {
        var t=new Table();
        var pr=new TableProperties(new TableWidth{Width=ContentWidth.ToString(),Type=TableWidthUnitValues.Dxa},new TableJustification{Val=TableRowAlignmentValues.Center},
          new TableBorders(new TopBorder{Val=BorderValues.Single,Size=4,Color="000000"},new LeftBorder{Val=BorderValues.Single,Size=4,Color="000000"},new BottomBorder{Val=BorderValues.Single,Size=4,Color="000000"},new RightBorder{Val=BorderValues.Single,Size=4,Color="000000"},new InsideHorizontalBorder{Val=BorderValues.Single,Size=4,Color="000000"},new InsideVerticalBorder{Val=BorderValues.Single,Size=4,Color="000000"}),
          new TableLayout{Type=TableLayoutValues.Fixed},new TableCellMarginDefault(new TopMargin{Width="100",Type=TableWidthUnitValues.Dxa},new TableCellLeftMargin{Width=100,Type=TableWidthValues.Dxa},new BottomMargin{Width="100",Type=TableWidthUnitValues.Dxa},new TableCellRightMargin{Width=100,Type=TableWidthValues.Dxa}));
        t.Append(pr);
        var grid=new TableGrid();var w=widths.Select(x=>ContentWidth*x/100).ToArray();w[^1]+=ContentWidth-w.Sum();
        foreach(int cw in w)grid.Append(new GridColumn{Width=cw.ToString()});t.Append(grid);
        var all=new List<string[]>{headers};all.AddRange(rows);
        for(int i=0;i<all.Count;i++) {
            var tr=new TableRow();var trPr=new TableRowProperties(new CantSplit());if(i==0)trPr.Append(new TableHeader());tr.Append(trPr);
            for(int j=0;j<headers.Length;j++) {
                var tc=new TableCell(new TableCellProperties(new TableCellWidth{Width=w[j].ToString(),Type=TableWidthUnitValues.Dxa},new TableCellVerticalAlignment{Val=TableVerticalAlignmentValues.Center}));
                var p=P(all[i][j],"TableText",(i==0||widths[j]<=20)?JustificationValues.Center:JustificationValues.Left,i==0,23);
                bool keepTable=rootConfig.TryGetProperty("keep_tables",out var keep) && keep.GetBoolean();
                if(i==0 || ((keepTable || keepTogether) && i<all.Count-1))p.ParagraphProperties!.InsertAfter(new KeepNext(),p.ParagraphProperties.FirstChild);
                tc.Append(p);tr.Append(tc);
            }
            t.Append(tr);
        }
        body.Append(t);
        body.Append(new Paragraph(new ParagraphProperties(new ParagraphStyleId{Val="TableText"},new SpacingBetweenLines{After="60",Line="60",LineRule=LineSpacingRuleValues.Exact})));
    }
    static void Heading(string title,int level,string bookmark,bool breakBefore=false,bool numbered=true) {
        var p=P(title,$"Heading{level+1}",bold:true);
        if(breakBefore)p.ParagraphProperties!.Append(new PageBreakBefore());
        if(numbered)p.ParagraphProperties!.Append(new NumberingProperties(new NumberingLevelReference{Val=level},new NumberingId{Val=1}));
        // SDK creates setters in schema order when used; sort direct properties before saving below.
        string id=(bookmarkId++).ToString();
        p.InsertAfter(new BookmarkStart{Id=id,Name=bookmark},p.ParagraphProperties);
        p.Append(new BookmarkEnd{Id=id});body.Append(p);
    }
    static void Toc() {
        CoverP("MỤC LỤC",28,true,0,180);
        for(int i=0;i<entries.Count;i++) {
            var e=entries[i];var p=P("",e.Level==0?"TOC1":"TOC2");
            p.ParagraphProperties!.Append(new Tabs(new TabStop{Val=TabStopValues.Right,Leader=TabStopLeaderCharValues.Dot,Position=ContentWidth}),new Indentation{Left=e.Level==0?"0":"360",FirstLine="0"});
            if(i==0)p.Append(new Run(RPr(),new FieldChar{FieldCharType=FieldCharValues.Begin}),new Run(RPr(),new FieldCode(" TOC \\o \"1-2\" \\h \\z \\u "){Space=SpaceProcessingModeValues.Preserve}),new Run(RPr(),new FieldChar{FieldCharType=FieldCharValues.Separate}));
            var link=new Hyperlink{Anchor=e.Bookmark,History=true};
            link.Append(Run(e.Title,22,e.Level==0),new Run(RPr(22),new TabChar()),Run(pages.GetValueOrDefault(e.Bookmark,"0"),22,e.Level==0));p.Append(link);
            if(i==entries.Count-1)p.Append(new Run(RPr(),new FieldChar{FieldCharType=FieldCharValues.End}));
            body.Append(p);
        }
    }
    static void Reorder(OpenXmlElement el) {
        // Order paragraph-property singletons per ECMA-376; other constructed elements already ordered.
        string[] tags={"pStyle","keepNext","keepLines","pageBreakBefore","framePr","widowControl","numPr","suppressLineNumbers","pBdr","shd","tabs","suppressAutoHyphens","kinsoku","wordWrap","overflowPunct","topLinePunct","autoSpaceDE","autoSpaceDN","bidi","adjustRightInd","snapToGrid","spacing","ind","contextualSpacing","mirrorIndents","suppressOverlap","jc","textDirection","textAlignment","outlineLvl","divId","cnfStyle","rPr","sectPr","pPrChange"};
        foreach(var pp in el.Descendants<ParagraphProperties>().ToList()) {
            var ordered=pp.ChildElements.OrderBy(x=>Array.IndexOf(tags,x.LocalName)).Select(x=>x.CloneNode(true)).ToArray();pp.RemoveAllChildren();pp.Append(ordered);
        }
    }
    static Paragraph CoverText(string text,int size=28,int before=0,int after=0,string style="Cover") {
        var pr=new ParagraphProperties(new ParagraphStyleId{Val=style},new KeepLines(),
            new SpacingBetweenLines{Before=before.ToString(),After=after.ToString(),Line="276",LineRule=LineSpacingRuleValues.Auto},
            new Indentation{Left="0",Right="0",FirstLine="0"},new Justification{Val=JustificationValues.Center});
        return new Paragraph(pr,Run(text,size,true));
    }
    static Paragraph CoverRecord(string label,string value,int before=0) {
        var pr=new ParagraphProperties(new ParagraphStyleId{Val="Cover"},new KeepNext(),new KeepLines(),
            new Tabs(new TabStop{Val=TabStopValues.Right,Position=3120},new TabStop{Val=TabStopValues.Left,Position=3480}),
            new SpacingBetweenLines{Before=before.ToString(),After="70",Line="276",LineRule=LineSpacingRuleValues.Auto},
            new Indentation{Left="0",Right="0",FirstLine="0"},new Justification{Val=JustificationValues.Left});
        return new Paragraph(pr,new Run(RPr(26,true),new TabChar()),Run(label,26,true),
            new Run(RPr(26,true),new TabChar()),Run(value,26,true));
    }
    static Paragraph CoverLogo(string path) {
        var part=main.AddImagePart(ImagePartType.Jpeg);
        using(var file=File.OpenRead(path))part.FeedData(file);
        const long dimension=1828800L;
        var drawing=new Drawing(new DW.Inline(
            new DW.Extent{Cx=dimension,Cy=dimension},
            new DW.EffectExtent{LeftEdge=0,TopEdge=0,RightEdge=0,BottomEdge=0},
            new DW.DocProperties{Id=1001U,Name="Logo Đại học Sài Gòn",Description="Logo Đại học Sài Gòn lấy từ trang bìa do nhóm cung cấp"},
            new DW.NonVisualGraphicFrameDrawingProperties(new A.GraphicFrameLocks{NoChangeAspect=true}),
            new A.Graphic(new A.GraphicData(new PIC.Picture(
                new PIC.NonVisualPictureProperties(new PIC.NonVisualDrawingProperties{Id=0U,Name="SGU-logo.jpg"},new PIC.NonVisualPictureDrawingProperties()),
                new PIC.BlipFill(new A.Blip{Embed=main.GetIdOfPart(part)},new A.Stretch(new A.FillRectangle())),
                new PIC.ShapeProperties(new A.Transform2D(new A.Offset{X=0L,Y=0L},new A.Extents{Cx=dimension,Cy=dimension}),
                    new A.PresetGeometry(new A.AdjustValueList()){Preset=A.ShapeTypeValues.Rectangle})))
                {Uri="http://schemas.openxmlformats.org/drawingml/2006/picture"}))
            {DistanceFromTop=0,DistanceFromBottom=0,DistanceFromLeft=0,DistanceFromRight=0});
        var paragraph=CoverText("",28,480,400);
        paragraph.RemoveAllChildren<Run>();paragraph.Append(new Run(drawing));
        return paragraph;
    }
    static void UpdateCover(string input,string output,string logo) {
        File.Copy(input,output,true);
        using var document=WordprocessingDocument.Open(output,true);
        main=document.MainDocumentPart!;body=main.Document!.Body!;
        var boundary=body.Elements<Paragraph>().First(p=>p.ParagraphProperties?.GetFirstChild<SectionProperties>()!=null);
        var section=boundary.ParagraphProperties!.GetFirstChild<SectionProperties>()!;
        while(body.FirstChild!=boundary)body.FirstChild!.Remove();
        section.GetFirstChild<PageSize>()!.Width=11906U;
        section.GetFirstChild<PageSize>()!.Height=16838U;
        var margin=section.GetFirstChild<PageMargin>()!;
        margin.Top=1440;margin.Bottom=1440;margin.Left=1440U;margin.Right=1440U;
        section.RemoveAllChildren<PageBorders>();
        section.InsertAfter(new PageBorders(
            new TopBorder{Val=BorderValues.Double,Size=6,Space=12,Color="000000"},
            new LeftBorder{Val=BorderValues.Double,Size=6,Space=12,Color="000000"},
            new BottomBorder{Val=BorderValues.Double,Size=6,Space=12,Color="000000"},
            new RightBorder{Val=BorderValues.Double,Size=6,Space=12,Color="000000"})
            {OffsetFrom=PageBorderOffsetValues.Text},margin);
        var cover=new OpenXmlElement[]{
            CoverText("TRƯỜNG ĐẠI HỌC SÀI GÒN"),
            CoverText("KHOA CÔNG NGHỆ THÔNG TIN",28,0,120),
            CoverLogo(logo),
            CoverText("BÁO CÁO KẾ HOẠCH ĐỀ TÀI",36,420,120,"Title"),
            CoverText("MÔN TRÍ TUỆ NHÂN TẠO NÂNG CAO",28,0,420),
            CoverText("TÊN ĐỀ TÀI",28,0,80),
            CoverText("XÂY DỰNG HỆ THỐNG CỜ VUA",32),
            CoverText("TÍCH HỢP TRÍ TUỆ NHÂN TẠO",32,0,480),
            CoverRecord("Nhóm báo cáo:","03"),
            CoverRecord("Giảng viên hướng dẫn:","Đỗ Như Tài"),
            CoverRecord("Danh sách thành viên:","Phạm Minh Dương (3123410065)",480),
            CoverRecord("","Phạm Tấn Đạt (3123410072)"),
            CoverRecord("","Đỗ Quốc Việt (3123410426)"),
            CoverRecord("","Nguyễn Xuân Đăng (3123410077)"),
            CoverText("Thành phố Hồ Chí Minh, tháng 11 năm 2026",26,1900)
        };
        foreach(var element in cover)body.InsertBefore(element,boundary);
        var members=new[]{
            (Name:"Phạm Minh Dương",Id:"3123410065"),
            (Name:"Phạm Tấn Đạt",Id:"3123410072"),
            (Name:"Đỗ Quốc Việt",Id:"3123410426"),
            (Name:"Nguyễn Xuân Đăng",Id:"3123410077")
        };
        var memberFields=body.Elements<Paragraph>().Where(p=>p.InnerText.StartsWith("Họ tên:")).ToList();
        if(memberFields.Count!=members.Length)throw new InvalidOperationException("Expected four member fields in the work allocation chapter.");
        for(int i=0;i<members.Length;i++) {
            var texts=memberFields[i].Descendants<Text>().ToList();
            texts[0].Text=$"Họ tên: {members[i].Name}    Mã số sinh viên: {members[i].Id}";
            foreach(var text in texts.Skip(1))text.Text="";
        }
        Reorder(main.Document);main.Document.Save();
        document.PackageProperties.Creator="Phạm Minh Dương; Phạm Tấn Đạt; Đỗ Quốc Việt; Nguyễn Xuân Đăng";
        var errors=new OpenXmlValidator(FileFormatVersions.Office2019).Validate(document).ToList();
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(output)!,"report-build/cover-update/validation.json"),
            JsonSerializer.Serialize(errors.Select(e=>new{e.Description,Path=e.Path?.XPath}),new JsonSerializerOptions{WriteIndented=true}));
        Console.WriteLine($"Updated cover: {output}; validation errors: {errors.Count}");
        if(errors.Count>0)throw new InvalidOperationException(string.Join("\n",errors.Select(e=>e.Description)));
    }
    static void ReplaceParagraphText(Paragraph paragraph,string text) {
        var first=paragraph.Descendants<Text>().FirstOrDefault();
        if(first==null)paragraph.Append(Run(text));
        else {
            first.Text=text;
            foreach(var other in paragraph.Descendants<Text>().Skip(1))other.Text="";
        }
    }
    static void RevisePlan(string input,string output,string editsPath,string? pageMapPath) {
        File.Copy(input,output,true);
        using var edits=JsonDocument.Parse(File.ReadAllText(editsPath));
        using var document=WordprocessingDocument.Open(output,true);
        main=document.MainDocumentPart!;body=main.Document!.Body!;
        foreach(var edit in edits.RootElement.GetProperty("replacements").EnumerateArray()) {
            string prefix=edit.GetProperty("starts_with").GetString()!;
            var matches=body.Elements<Paragraph>().Where(p=>p.InnerText.StartsWith(prefix,StringComparison.Ordinal)).ToList();
            if(matches.Count!=1)throw new InvalidOperationException($"Expected one paragraph: {prefix}; found {matches.Count}");
            var original=matches[0];var replacements=edit.GetProperty("paragraphs").EnumerateArray().Select(x=>x.GetString()!).ToArray();
            if(replacements.Length==0) {original.Remove();continue;}
            ReplaceParagraphText(original,replacements[0]);
            OpenXmlElement previous=original;
            foreach(var text in replacements.Skip(1)) {
                var next=(Paragraph)original.CloneNode(true);ReplaceParagraphText(next,text);
                body.InsertAfter(next,previous);previous=next;
            }
        }
        foreach(var addition in edits.RootElement.GetProperty("table_additions").EnumerateArray()) {
            string header=addition.GetProperty("header").GetString()!;
            var table=body.Elements<Table>().Single(t=>t.Elements<TableRow>().First().Elements<TableCell>().First().InnerText==header);
            var after=table.Elements<TableRow>().Single(r=>r.Elements<TableCell>().First().InnerText==addition.GetProperty("after").GetString());
            var row=(TableRow)after.CloneNode(true);
            var cells=addition.GetProperty("cells").EnumerateArray().Select(x=>x.GetString()!).ToArray();
            int i=0;foreach(var cell in row.Elements<TableCell>())ReplaceParagraphText(cell.Elements<Paragraph>().Single(),cells[i++]);
            table.InsertAfter(row,after);
        }
        foreach(var edit in edits.RootElement.GetProperty("cell_replacements").EnumerateArray()) {
            string old=edit.GetProperty("old").GetString()!;
            var cell=body.Descendants<TableCell>().Single(c=>c.InnerText==old);
            ReplaceParagraphText(cell.Elements<Paragraph>().Single(),edit.GetProperty("new").GetString()!);
        }
        foreach(var text in body.Descendants<Text>())
            text.Text=Regex.Replace(text.Text,@"\[(7|8)\]",m=>m.Groups[1].Value=="7"?"[6]":"[7]");
        var tocEntries=JsonSerializer.Deserialize<List<Entry>>(File.ReadAllText(Path.Combine(Path.GetDirectoryName(editsPath)!,"../revision-v2/toc-entries.json")))!;
        foreach(var entry in tocEntries) {
            var bookmark=body.Descendants<BookmarkStart>().Single(b=>b.Name==entry.Bookmark);
            var heading=bookmark.Ancestors<Paragraph>().Single();
            ReplaceParagraphText(heading,entry.Title);
            heading.ParagraphProperties!.RemoveAllChildren<NumberingProperties>();
        }
        if(pageMapPath!=null) {
            var pageMap=JsonSerializer.Deserialize<Dictionary<string,string>>(File.ReadAllText(pageMapPath))!;
            foreach(var link in body.Descendants<Hyperlink>()) {
                if(link.Anchor!=null && pageMap.TryGetValue(link.Anchor.Value!,out var number))link.Descendants<Text>().Last().Text=number;
            }
        }
        Reorder(main.Document);main.Document.Save();
        var errors=new OpenXmlValidator(FileFormatVersions.Office2019).Validate(document).ToList();
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(editsPath)!,"validation.json"),
            JsonSerializer.Serialize(errors.Select(e=>new{e.Description,Path=e.Path?.XPath}),new JsonSerializerOptions{WriteIndented=true}));
        Console.WriteLine($"Revised plan: {output}; validation errors: {errors.Count}");
        if(errors.Count>0)throw new InvalidOperationException(string.Join("\n",errors.Select(e=>e.Description)));
    }
    static void RewritePlan(string input,string output,string manuscript,string? pageMapPath) {
        File.Copy(input,output,true);
        using var content=JsonDocument.Parse(File.ReadAllText(manuscript));rootConfig=content.RootElement;
        if(pageMapPath!=null)pages=JsonSerializer.Deserialize<Dictionary<string,string>>(File.ReadAllText(pageMapPath))!;
        using var document=WordprocessingDocument.Open(output,true);
        main=document.MainDocumentPart!;body=main.Document!.Body!;
        var sectionEnds=body.Elements<Paragraph>().Where(p=>p.ParagraphProperties?.GetFirstChild<SectionProperties>()!=null).ToList();
        if(sectionEnds.Count!=2)throw new InvalidOperationException("Expected cover, TOC and body sections.");
        var boundary=sectionEnds[0];
        var tocSection=(SectionProperties)sectionEnds[1].ParagraphProperties!.GetFirstChild<SectionProperties>()!.CloneNode(true);
        var bodySection=(SectionProperties)body.Elements<SectionProperties>().Single().CloneNode(true);
        while(boundary.NextSibling()!=null)boundary.NextSibling()!.Remove();
        int ch=0;
        foreach(var chapter in rootConfig.GetProperty("chapters").EnumerateArray()) {
            ch++;entries.Add(new(){Title=$"CHƯƠNG {ch} {chapter.GetProperty("title").GetString()}",Bookmark=$"ch{ch}",Level=0});int s=0;
            foreach(var sec in chapter.GetProperty("sections").EnumerateArray()) {
                s++;entries.Add(new(){Title=$"{ch}.{s} {sec.GetProperty("title").GetString()}",Bookmark=$"s{ch}_{s}",Level=1});
            }
        }
        entries.Add(new(){Title="TÀI LIỆU THAM KHẢO",Bookmark="references",Level=0});
        Toc();EndSection(tocSection);ch=0;
        foreach(var chapter in rootConfig.GetProperty("chapters").EnumerateArray()) {
            ch++;Heading($"CHƯƠNG {ch} {chapter.GetProperty("title").GetString()}",0,$"ch{ch}",ch>1,false);int s=0;
            foreach(var sec in chapter.GetProperty("sections").EnumerateArray()) {
                s++;bool breakBefore=sec.TryGetProperty("page_break_before",out var pb) && pb.GetBoolean();
                Heading($"{ch}.{s} {sec.GetProperty("title").GetString()}",1,$"s{ch}_{s}",breakBefore,false);
                var paragraphs=sec.GetProperty("paragraphs").EnumerateArray().Select(x=>x.GetString()!).ToList();
                int beforeTable=sec.TryGetProperty("table_after_paragraph",out var tp)?tp.GetInt32():paragraphs.Count;
                foreach(var paragraph in paragraphs.Take(beforeTable))AddP(paragraph);
                if(sec.TryGetProperty("table",out var table)) {
                    var caption=P(table.GetProperty("caption").GetString()!,"Caption",bold:true);
                    caption.ParagraphProperties!.Append(new KeepNext());body.Append(caption);
                    Table(table.GetProperty("headers").EnumerateArray().Select(x=>x.GetString()!).ToArray(),
                        table.GetProperty("rows").EnumerateArray().Select(row=>row.EnumerateArray().Select(x=>x.GetString()!).ToArray()).ToArray(),
                        table.GetProperty("widths").EnumerateArray().Select(x=>x.GetInt32()).ToArray(),
                        table.TryGetProperty("keep_together",out var kt) && kt.GetBoolean());
                }
                foreach(var paragraph in paragraphs.Skip(beforeTable))AddP(paragraph);
                if(sec.TryGetProperty("afterparagraphs",out var after))foreach(var paragraph in after.EnumerateArray())AddP(paragraph.GetString()!);
            }
        }
        Heading("TÀI LIỆU THAM KHẢO",0,"references",true,false);
        foreach(var reference in rootConfig.GetProperty("references").EnumerateArray())body.Append(P(reference.GetString()!,"Bibliography"));
        body.Append(bodySection);Reorder(main.Document);main.Document.Save();
        var errors=new OpenXmlValidator(FileFormatVersions.Office2019).Validate(document).ToList();
        string dir=Path.GetDirectoryName(manuscript)!;
        File.WriteAllText(Path.Combine(dir,"validation.json"),JsonSerializer.Serialize(errors.Select(e=>new{e.Description,Path=e.Path?.XPath}),new JsonSerializerOptions{WriteIndented=true}));
        File.WriteAllText(Path.Combine(dir,"toc-entries.json"),JsonSerializer.Serialize(entries,new JsonSerializerOptions{WriteIndented=true}));
        Console.WriteLine($"Rewritten plan: {output}; headings: {entries.Count}; validation errors: {errors.Count}");
        if(errors.Count>0)throw new InvalidOperationException(string.Join("\n",errors.Select(e=>e.Description)));
    }
    public static void Main(string[] args) {
        if(args.Length>0 && args[0]=="--rewrite-plan") {
            RewritePlan(args[1],args[2],args[3],args.Length>4?args[4]:null);return;
        }
        if(args.Length>0 && args[0]=="--revise-plan") {
            RevisePlan(args[1],args[2],args[3],args.Length>4?args[4]:null);return;
        }
        if(args.Length>0 && args[0]=="--update-cover") {
            UpdateCover(args[1],args[2],args[3]);return;
        }
        string src=args[0], output=args[1];
        if(args.Length>2 && File.Exists(args[2]))pages=JsonSerializer.Deserialize<Dictionary<string,string>>(File.ReadAllText(args[2]))!;
        using var content=JsonDocument.Parse(File.ReadAllText(src));
        rootConfig=content.RootElement;
        bool personalPages=!rootConfig.TryGetProperty("personal_pages",out var ppConfig) || ppConfig.GetBoolean();
        int ch=0;
        foreach(var c in content.RootElement.GetProperty("chapters").EnumerateArray()) {
            ch++;entries.Add(new(){Title=$"CHƯƠNG {ch} {c.GetProperty("title").GetString()}",Bookmark=$"ch{ch}",Level=0});int s=0;
            foreach(var sec in c.GetProperty("sections").EnumerateArray()) {s++;entries.Add(new(){Title=$"{ch}.{s} {sec.GetProperty("title").GetString()}",Bookmark=$"s{ch}_{s}",Level=1});}
        }
        entries.Add(new(){Title="TÀI LIỆU THAM KHẢO",Bookmark="references",Level=0});
        using(var doc=WordprocessingDocument.Create(output,WordprocessingDocumentType.Document)) {
            main=doc.AddMainDocumentPart();main.Document=new Document(new Body());body=main.Document.Body!;
            Styles();Numbering();
            var settings=main.AddNewPart<DocumentSettingsPart>();settings.Settings=new Settings(new UpdateFieldsOnOpen{Val=false});
            string blankHead=Header(""),blankFoot=Footer(false),pageFoot=Footer(true),running=Header(ConfigText("running_header", "KẾ HOẠCH ĐỀ TÀI CỜ VUA AI"));
            Cover();EndSection(Section(blankHead,blankFoot));
            Toc();EndSection(Section(blankHead,pageFoot,NumberFormatValues.LowerRoman,true));
            ch=0;
            foreach(var c in content.RootElement.GetProperty("chapters").EnumerateArray()) {
                ch++;Heading(c.GetProperty("title").GetString()!,0,$"ch{ch}",ch>1);int s=0;
                foreach(var sec in c.GetProperty("sections").EnumerateArray()) {
                    s++;Heading(sec.GetProperty("title").GetString()!,1,$"s{ch}_{s}",personalPages && ch==6 && s>=3 && s<=5);
                    if(sec.TryGetProperty("paragraphs",out var pars))foreach(var p in pars.EnumerateArray())AddP(p.GetString()!);
                    if(sec.TryGetProperty("table",out var t)) {
                        var cap=P(t.GetProperty("caption").GetString()!,"Caption",bold:true);cap.ParagraphProperties!.Append(new KeepNext());body.Append(cap);
                        Table(t.GetProperty("headers").EnumerateArray().Select(x=>x.GetString()!).ToArray(),t.GetProperty("rows").EnumerateArray().Select(r=>r.EnumerateArray().Select(x=>x.GetString()!).ToArray()).ToArray(),t.GetProperty("widths").EnumerateArray().Select(x=>x.GetInt32()).ToArray());
                    }
                    if(sec.TryGetProperty("codeblocks",out var blocks))foreach(var block in blocks.EnumerateArray()) {
                        var cap=P(block.GetProperty("caption").GetString()!,"Caption",bold:true);cap.ParagraphProperties!.Append(new KeepNext());body.Append(cap);
                        var lines=block.GetProperty("text").GetString()!.Replace("\r", "").Trim('\n').Split('\n');
                        for(int line=0;line<lines.Length;line++) {
                            var code=P(lines[line],"Code",size:22);
                            code.ParagraphProperties!.Append(new KeepLines(),new Indentation{Left="180",FirstLine="0"});
                            if(line<lines.Length-1)code.ParagraphProperties.Append(new KeepNext());
                            body.Append(code);
                        }
                        body.Append(new Paragraph(new ParagraphProperties(new ParagraphStyleId{Val="TableText"},new SpacingBetweenLines{After="60",Line="60",LineRule=LineSpacingRuleValues.Exact})));
                    }
                    if(sec.TryGetProperty("afterparagraphs",out var after))foreach(var p in after.EnumerateArray())AddP(p.GetString()!);
                }
            }
            Heading("TÀI LIỆU THAM KHẢO",0,"references",true,false);
            foreach(var r in content.RootElement.GetProperty("references").EnumerateArray())body.Append(P(r.GetString()!,"Bibliography"));
            body.Append(Section(running,pageFoot,NumberFormatValues.Decimal,true));
            Reorder(main.Document);
            main.Document.Save();
            doc.PackageProperties.Title=ConfigText("document_title", "Kế hoạch thực hiện đề tài Cờ Vua tích hợp trí tuệ nhân tạo");
            doc.PackageProperties.Subject=ConfigText("document_subject", "Khảo sát đề tài, dữ liệu, độ đo, kế hoạch triển khai và phân công bốn thành viên");
            doc.PackageProperties.Creator="Nhóm thực hiện đề tài";
            doc.PackageProperties.Description=ConfigText("document_description", "Báo cáo kế hoạch học phần Trí tuệ nhân tạo nâng cao");
            var errors=new OpenXmlValidator(FileFormatVersions.Office2019).Validate(doc).ToList();
            File.WriteAllText(Path.Combine(Path.GetDirectoryName(src)!,"validation.json"),JsonSerializer.Serialize(errors.Select(e=>new{e.Description,Path=e.Path?.XPath}),new JsonSerializerOptions{WriteIndented=true}));
            Console.WriteLine($"DOCX: {output}\nHeadings: {entries.Count}; validation errors: {errors.Count}");
            if(errors.Count>0)foreach(var e in errors.Take(12))Console.WriteLine($"{e.Description} :: {e.Path?.XPath}");
        }
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(src)!,"toc-entries.json"),JsonSerializer.Serialize(entries,new JsonSerializerOptions{WriteIndented=true}));
    }
}
