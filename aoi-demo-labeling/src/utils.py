def xyxy_to_xywh(a): 
    x1 = min(a[0], a[2])
    y1 = min(a[1], a[3])
    x2 = max(a[0], a[2])
    y2 = max(a[1], a[3])    
    return [x1,y1,x2-x1+1,y2-y1+1]


def xywh_xyxy(a): 
    return [a[0],a[1],a[0]+a[2]-1,a[1]+a[3]-1]