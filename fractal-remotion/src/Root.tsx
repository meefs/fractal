import "./index.css";
import { Composition } from "remotion";
import { FractalRelease, FractalReleaseMobile } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="FractalRelease"
        component={FractalRelease}
        durationInFrames={618}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="FractalReleaseMobile"
        component={FractalReleaseMobile}
        durationInFrames={618}
        fps={30}
        width={1080}
        height={1350}
      />
    </>
  );
};
