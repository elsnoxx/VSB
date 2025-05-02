import React, { useEffect, useState } from "react";
import { IonList, IonItem, IonLabel, IonInfiniteScroll, IonInfiniteScrollContent } from "@ionic/react";

type Waypoint = {
  name: string;
  date: string;
};

const PAGE_SIZE = 10;

const WaypointsList: React.FC = () => {
  const [waypoints, setWaypoints] = useState<Waypoint[]>([]);
  const [displayed, setDisplayed] = useState<Waypoint[]>([]);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    const data = JSON.parse(localStorage.getItem("waypoints") || "[]");
    setWaypoints(data);
    setDisplayed(data.slice(0, PAGE_SIZE));
    setHasMore(data.length > PAGE_SIZE);
  }, []);

  const loadMore = (e: CustomEvent<void>) => {
    const next = displayed.length + PAGE_SIZE;
    setDisplayed(waypoints.slice(0, next));
    setHasMore(next < waypoints.length);
    (e.target as HTMLIonInfiniteScrollElement).complete();
  };

  return (
    <>
      <IonList>
        {displayed.map((wp, idx) => (
          <IonItem key={idx}>
            <IonLabel>
              <b>{wp.name}</b>
              <div style={{ fontSize: "0.9em", color: "#888" }}>
                {new Date(wp.date).toLocaleString()}
              </div>
            </IonLabel>
          </IonItem>
        ))}
      </IonList>
      <IonInfiniteScroll
        threshold="100px"
        disabled={!hasMore}
        onIonInfinite={loadMore}
      >
        <IonInfiniteScrollContent loadingText="Načítám další body..." />
      </IonInfiniteScroll>
    </>
  );
};

export default WaypointsList;