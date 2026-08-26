import React, { useEffect, useMemo, useState } from "react";
import { Button, Input, Select, Space, Spin, Table, Tag, Typography, message } from "antd";
import { getProductArchive, type ProductArchiveItem } from "../api";
import { productNameCategories, type ProductNameCategory } from "./productDisplay";

const pct = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
type VolatilityBand = "相对低波" | "中波" | "相对高波" | "未分层";

function volatilityBand(value: number | null | undefined, low: number | undefined, high: number | undefined): VolatilityBand {
  if (value == null || low == null || high == null) return "未分层";
  if (value <= low) return "相对低波";
  return value <= high ? "中波" : "相对高波";
}

interface ArchivePanelProps {
  onOpenProduct?: (product: ProductArchiveItem) => void;
  /** 对比栏选中的产品（上限两只），与对话用的选中产品完全独立。 */
  compareIds?: string[];
  compareNames?: Record<string, string>;
  onCompareChange?: (ids: string[], names: Record<string, string>) => void;
  onToggleCompare?: (productId: string, name?: string) => void;
  onOpenCompare?: () => void;
}

export default function ProductArchivePanel({ onOpenProduct, compareIds = [], compareNames = {}, onCompareChange, onToggleCompare, onOpenCompare }: ArchivePanelProps): React.JSX.Element {
  const [items, setItems] = useState<ProductArchiveItem[]>([]); const [loading, setLoading] = useState(true); const [search,setSearch]=useState(""); const [strategy,setStrategy]=useState<string>(); const [category, setCategory] = useState<ProductNameCategory>(); const [volatility, setVolatility] = useState<VolatilityBand>(); const [dataStatus, setDataStatus] = useState<ProductArchiveItem["data_status"]>();
  useEffect(()=>{ void getProductArchive().then((result)=>setItems(result.items)).finally(()=>setLoading(false)); },[]);
  const strategies=useMemo(()=>[...new Set(items.map(x=>x.strategy).filter(Boolean))] as string[],[items]);
  const categories=useMemo(()=>[...new Set(items.flatMap((item)=>productNameCategories(item.name,item.strategy)))],[items]);
  const cutoffs=useMemo(()=>{ const values=items.map((item)=>item.annual_volatility).filter((value):value is number=>Number.isFinite(value)).sort((left,right)=>left-right); return values.length ? [values[Math.floor((values.length-1)/3)],values[Math.floor((values.length-1)*2/3)]] as const : [undefined,undefined] as const; },[items]);
  const data=useMemo(()=>items.filter((item)=>{
    const itemCategories=productNameCategories(item.name,item.strategy);
    return (!strategy||item.strategy===strategy)&&(!category||itemCategories.includes(category))&&(!volatility||volatilityBand(item.annual_volatility,...cutoffs)===volatility)&&(!dataStatus||item.data_status===dataStatus)&&`${item.name} ${item.manager??""}`.includes(search);
  }),[items,search,strategy,category,volatility,dataStatus,cutoffs]);
  return <><Space style={{marginBottom:10}} wrap><Input placeholder="搜索产品或管理人" value={search} onChange={e=>setSearch(e.target.value)} /><Select allowClear placeholder="策略" style={{width:130}} value={strategy} onChange={setStrategy} options={strategies.map(value=>({value}))}/><Select allowClear placeholder="名称标签" style={{width:120}} value={category} onChange={setCategory} options={categories.map(value=>({value}))}/><Select allowClear placeholder="波动分层" style={{width:120}} value={volatility} onChange={setVolatility} options={["相对低波","中波","相对高波"].map(value=>({value}))}/><Select allowClear placeholder="数据状态" style={{width:120}} value={dataStatus} onChange={setDataStatus} options={["正常","高回撤"].map(value=>({value}))}/><Typography.Text type="secondary">{data.length} 个产品</Typography.Text></Space>{onCompareChange && compareIds.length>0&&<div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:12,padding:"8px 12px",marginBottom:10,border:"1px solid var(--serif-border)",borderRadius:8,background:"var(--serif-card)"}}><Space size={8} wrap><Typography.Text strong>对比</Typography.Text>{compareIds.map((id)=><Tag key={id} closable onClose={(event)=>{event.preventDefault();onToggleCompare?.(id);}}>{compareNames[id]??"未命名产品"}</Tag>)}{compareIds.length===1&&<Typography.Text type="secondary">再勾选一只产品</Typography.Text>}</Space><Button size="small" type="primary" disabled={compareIds.length<2} onClick={onOpenCompare}>查看对比</Button></div>}{loading?<Spin/>:<Table size="small" rowKey="product_id" scroll={{x:1450}} dataSource={data} pagination={{pageSize:20,showSizeChanger:false}} rowSelection={onCompareChange?{selectedRowKeys:compareIds,onChange:(keys,rows)=>{const ids=keys.map(String);if(ids.length>2){message.warning("对比最多选择两只产品，请先取消一只。");return;}const names:Record<string,string>={};for(const row of rows) names[row.product_id]=row.name;onCompareChange(ids,names);}}:undefined} columns={[{title:"排名",dataIndex:"rank",width:60,sorter:(a,b)=>(a.rank??9999)-(b.rank??9999)},{title:"产品",dataIndex:"name",render:(name,row)=><><Button type="link" size="small" style={{padding:0,height:"auto"}} onClick={()=>onOpenProduct?.(row)}>{name}</Button><Typography.Text type="secondary"> · {row.manager??"—"}</Typography.Text></>},{title:"名称标签",render:(_v,row)=>{const labels=productNameCategories(row.name,row.strategy);return labels.length?<>{labels.map(category=><Tag key={category} color={category==="CTA"?"blue":category==="股票"?"green":category==="波动率"?"purple":"default"}>{category}</Tag>)}</>:"—";}},{title:"策略",dataIndex:"strategy",render:v=><Tag>{v??"未分类"}</Tag>},{title:"数据状态",dataIndex:"data_status",render:value=><Tag color={value==="高回撤"?"red":"green"}>{value}</Tag>},{title:"质量分",dataIndex:"quality_score",sorter:(a,b)=>(a.quality_score??-1)-(b.quality_score??-1),render:v=>v?.toFixed(1)??"—"},{title:"置信分",dataIndex:"confidence_score",render:v=>v?.toFixed(1)??"—"},{title:"归因质量分",dataIndex:"attribution_quality_score",render:v=>v?.toFixed(1)??"—"},{title:"R²",dataIndex:"r_squared",render:v=>v?.toFixed(2)??"—"},{title:"样本外 R²",dataIndex:"oos_r_squared",render:v=>v?.toFixed(2)??"—"},{title:"年化收益",dataIndex:"annual_return",sorter:(a,b)=>(a.annual_return??-9)-(b.annual_return??-9),render:pct},{title:"年化波动",dataIndex:"annual_volatility",sorter:(a,b)=>a.annual_volatility-b.annual_volatility,render:pct},{title:"波动分层",render:(_v,row)=><Tag color={volatilityBand(row.annual_volatility,...cutoffs)==="相对高波"?"red":volatilityBand(row.annual_volatility,...cutoffs)==="相对低波"?"green":"default"}>{volatilityBand(row.annual_volatility,...cutoffs)}</Tag>},{title:"Sharpe",dataIndex:"sharpe",sorter:(a,b)=>(a.sharpe??-9)-(b.sharpe??-9),render:v=>v?.toFixed(2)??"—"},{title:"最大回撤",dataIndex:"maximum_drawdown",sorter:(a,b)=>a.maximum_drawdown-b.maximum_drawdown,render:pct}]} />}</>;
}
