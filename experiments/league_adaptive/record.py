"""A few labelled native-frame videos, without course packaging utilities."""
import argparse,json,math,subprocess,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.league_adaptive.runtime import load_actor,write,setup
from experiments.league_adaptive.environment import FightEnv,family,scheduled_style


def input_history_analysis(definition,decisions,cap=120):
    """Issued-key candidates for visual review, never move-success labels."""
    from experiments.league_adaptive.characters import advance_input_durations,PUNCHES,KICKS
    from experiments.league_adaptive.runtime import compiled
    commands=compiled(definition['actions']);counts=np.zeros(3,np.uint8);maximum=counts.copy();candidates=[]
    for row in decisions:
        command=commands[row['physical_action']];types=[]
        if definition.get('character')=='Guile':
            front='RIGHT' if row['opponent_x_before']>=row['self_x_before'] else 'LEFT'
            back_index=0 if front=='RIGHT' else 1
            if counts[back_index]>=60 and any(front in keys and set(keys)&set(PUNCHES) for keys in command):
                types.append('coordinate-facing_horizontal_release_input')
            if counts[2]>=60 and any('UP' in keys and set(keys)&set(KICKS) for keys in command):
                types.append('vertical_release_input')
        if types:
            candidates.append(dict(frame=row['frame'],decision_start_frame=row['decision_start_frame'],action=row['action'],
                prior_issued_counts=dict(zip(('LEFT','RIGHT','DOWN'),map(int,counts))),input_types=types,
                self_y_before=row['self_y_before'],opponent_y_before=row['opponent_y_before'],
                following120frame_damage_dealt=sum(r['damage_dealt_hp'] for r in decisions if row['frame']<=r['frame']<row['frame']+120),
                following120frame_damage_taken=sum(r['damage_taken_hp'] for r in decisions if row['frame']<=r['frame']<row['frame']+120)))
        count=min(6,row['frame']-row['decision_start_frame'])
        counts=advance_input_durations(counts,command[:count],cap);maximum=np.maximum(maximum,counts)
    return dict(counter_cap_native_frames=cap,maximum_at_decision_boundaries=dict(zip(('LEFT','RIGHT','DOWN'),map(int,maximum))),
        candidate_release_inputs=candidates,
        scope='Capped own issued-key durations, not native charge readiness. Coordinate facing can differ from sprite facing during a cross. Inputs and following damage require visual attribution; they do not establish a special activation, causal benefit or an oracle action.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--agent',required=True);p.add_argument('--opponent');p.add_argument('--opponent-pool');p.add_argument('--opponent-id')
    p.add_argument('--cpu-character');p.add_argument('--cpu-stars',type=int,choices=range(1,9),default=8)
    p.add_argument('--player',type=int,choices=(1,2),default=1);p.add_argument('--offset',type=int,default=0);p.add_argument('--out',required=True)
    p.add_argument('--post-match-seconds',type=float,default=1.0,
                   help='Continue native animation with neutral controller inputs after the scored terminal frame.')
    p.add_argument('--label',default='learned response');p.add_argument('--opponent-label',default='frozen Counter-Ryu');a=p.parse_args();setup()
    if not math.isfinite(a.post_match_seconds) or a.post_match_seconds<0:
        raise ValueError('Post-match recording time must be finite and nonnegative')
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    if sum(bool(x) for x in (a.opponent,a.cpu_character,a.opponent_pool))!=1:raise ValueError('Specify one frozen opponent, native CPU character or frozen pool entry')
    if bool(a.opponent_pool)!=bool(a.opponent_id):raise ValueError('Select an exact opponent ID from the supplied pool')
    if a.cpu_character and a.player!=1:raise ValueError('Native single-player CPU uses learner controller P1')
    rival=dict(id=a.opponent_label,kind='agent',path=str(Path(a.opponent).resolve())) if a.opponent else dict(id=f'CPU_{a.cpu_character}_{a.cpu_stars}',kind='cpu',character=a.cpu_character,stars=a.cpu_stars)
    if a.opponent_pool:
        rival=next(q for q in json.loads(Path(a.opponent_pool).read_text()) if q['id']==a.opponent_id)
        if a.opponent_label=='frozen Counter-Ryu':a.opponent_label=a.opponent_id.replace('_',' ')
    if rival['kind']=='cpu':
        if a.player!=1:raise ValueError('Native single-player CPU uses learner controller P1')
        a.cpu_character=rival.get('character','Ryu');a.cpu_stars=rival['stars']
    if a.cpu_character:a.opponent_label=f'CPU {a.cpu_character} {a.cpu_stars}stars'
    out.mkdir(parents=True);actor=load_actor(a.agent);env=FightEnv(actor.definition,[rival])
    recorder=None;decisions=[]
    try:
        obs,_=env.reset(options=dict(player=a.player,offset=a.offset));actor.reset();raw_video=out/'native_recording.mp4'
        recording_start_frame=env.game.elapsed;written_frames=0;terminal_info=None
        recorder=cv2.VideoWriter(str(raw_video),cv2.VideoWriter_fourcc(*'mp4v'),env.game.fps,(640,568))
        if not recorder.isOpened():raise RuntimeError('Cannot create video')
        def frame(rgb,raw,k1,k2,n):
            nonlocal written_frames
            image=np.zeros((568,640,3),np.uint8);image[:448]=cv2.cvtColor(cv2.resize(rgb,(640,448),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR)
            skill=actor.definition['option_names'][int(actor.last_skill_indices[0])] if hasattr(actor,'last_skill_indices') else 'direct physical policy'
            if hasattr(actor,'last_neural_route_probabilities'):skill='neural weights '+' '.join(f'{i}:{v:.2f}' for i,v in enumerate(actor.last_neural_route_probabilities[0]))
            if hasattr(actor,'last_residual_probability'):skill+=f' | direct mass {float(actor.last_residual_probability[0,0]):.2f}'
            if hasattr(actor.policy,'_history_remaining') and actor.policy._history_remaining is not None:
                skill+=f' | pending input calls {int(actor.policy._history_remaining[0])}'
            phase=scheduled_style(rival,env.controller.count)
            right_input='native CPU' if a.cpu_character else '+'.join(k2) or 'neutral'
            own_x=raw[f'p{a.player}_x'];other_x=raw[f'p{3-a.player}_x'];relative='LEFT' if own_x<other_x else 'RIGHT' if own_x>other_x else 'SAME X'
            lines=[f'{a.label} P{a.player} [{relative}] vs {a.opponent_label} P{3-a.player}',
                f'P1 HP={raw["p1_hp"]} | P2 HP={raw["p2_hp"]} | native frame {n}',
                f'Skill: {skill}'+(f' | Opponent: {phase}' if phase else ''),f'P1 {"+".join(k1) or "neutral"} | P2 {right_input}']
            if terminal_info is not None:
                after=(n-terminal_info['native_frames'])/env.game.fps
                ending='KO' if terminal_info['end_reason']=='signed_hp_ko' else 'MATCH'
                lines[2]=f'POST-{ending} +{after:.2f}s | Final result: {terminal_info["result"].upper()}'
            for index,line in enumerate(lines):cv2.putText(image,line,(8,472+index*24),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
            recorder.write(image)
            written_frames+=1
        env.game.on_frame=frame
        while not env.ended:
            before=dict(env.game.raw);start_frame=env.game.elapsed
            action=int(actor.predict(obs,True)[0]);obs,_,_,_,info=env.step(action)
            raw=env.game.raw;p=a.player;q=3-p
            row=dict(frame=info['native_frames'],action=actor.actions[action]['name'],family=family(actor.actions[action]['name']),self_hp=info['self_hp'],opponent_hp=info['opponent_hp'],
                self_x=raw[f'p{p}_x'],self_y=raw[f'p{p}_y'],opponent_x=raw[f'p{q}_x'],opponent_y=raw[f'p{q}_y'],
                decision_start_frame=start_frame,physical_action=action,
                self_x_before=before[f'p{p}_x'],self_y_before=before[f'p{p}_y'],
                opponent_x_before=before[f'p{q}_x'],opponent_y_before=before[f'p{q}_y'],
                self_hp_before=before[f'p{p}_hp'],opponent_hp_before=before[f'p{q}_hp'],
                damage_dealt_hp=info['damage_dealt'],damage_taken_hp=info['damage_taken'])
            if hasattr(actor.policy,'last_context_q_values'):
                values=actor.policy.last_context_q_values[0].detach().cpu().numpy()
                order=np.argsort(values)
                row.update(selected_q=float(values[action]),q_top_two_gap=float(values[order[-1]]-values[order[-2]]),
                           q_runner_up_action=int(order[-2]))
            if getattr(actor.policy,'input_history',None) is not None:
                row['issued_input_durations_before']={key:int(value) for key,value in
                    zip(('LEFT','RIGHT','DOWN'),actor.policy.last_input_durations_before[0])}
            if hasattr(actor,'last_skill_indices'):row['selected_skill']=actor.definition['option_names'][int(actor.last_skill_indices[0])]
            if hasattr(actor,'last_neural_route_probabilities'):
                row['neural_route_probabilities']=actor.last_neural_route_probabilities[0].tolist()
                row['neural_proposed_inputs']=[actor.actions[int(i)]['name'] for i in actor.last_neural_proposals[0]]
            if hasattr(actor,'last_residual_probability'):row['direct_action_probability_mass']=float(actor.last_residual_probability[0,0])
            if hasattr(actor.policy,'last_option_starts'):
                row['new_input_option']=bool(actor.policy.last_option_starts[0])
                row['chosen_duration_calls']=int(actor.policy.last_duration_choices[0]) if row['new_input_option'] else None
                row['pending_input_calls']=int(actor.policy._history_remaining[0])
            phase=scheduled_style(rival,env.controller.count)
            if phase is not None:row['opponent_style']=phase
            decisions.append(row)
        # Freeze the scored outcome. Tail frames are animation only: no policy
        # calls, environment rewards, training transitions or further matches.
        terminal_info=dict(info)
        post_match_frames=math.ceil(a.post_match_seconds*env.game.fps)
        tail_visible_changes=0
        for _ in range(post_match_frames):
            previous_rgb=env.game.rgb.copy()
            env.game.step_keys()
            tail_visible_changes+=int(not np.array_equal(previous_rgb,env.game.rgb))
        recorder.release();recorder=None;video=out/'replay.mp4'
        import imageio_ffmpeg
        binary=Path(imageio_ffmpeg.get_ffmpeg_exe())
        subprocess.run([str(binary),'-loglevel','error','-y','-i',str(raw_video),'-c:v','libx264','-threads','1','-preset','veryfast','-crf','24','-pix_fmt','yuv420p',str(video)],check=True)
        raw_video.unlink();write(out/'decisions.json',decisions)
        capture=cv2.VideoCapture(str(video));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count!=written_frames:
            capture.release()
            raise RuntimeError('Encoded video did not retain all native match and post-match frames')
        write(out/'summary.json',dict(agent=str(Path(a.agent).resolve()),opponent=rival,player=a.player,offset=a.offset,
            **{k:info[k] for k in ('result','self_hp','opponent_hp','native_frames','end_reason')},video=str(video.resolve()),
            native_fps=env.game.fps,elapsed_seconds=info['native_frames']/env.game.fps,
            recorded_match_native_frames=info['native_frames']-recording_start_frame,
            post_match_requested_seconds=a.post_match_seconds,post_match_frames=post_match_frames,
            post_match_seconds=post_match_frames/env.game.fps,post_match_visible_changes=tail_visible_changes,
            post_match_inputs='Neutral controller inputs; native CPU remains game-controlled.',
            video_frames=count,video_seconds=count/env.game.fps,
            input_history_analysis=input_history_analysis(actor.definition,decisions),
            motion_scope='Decision endpoints include possible knockback. Inputs and endpoint motion alone do not establish that a movement caused the full-game result.',
            scope='Frozen learned-model replay for qualitative inspection; does not add an independent victory.'))
        for index,fraction in enumerate((.2,.5,.8)):
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(count*fraction));ok,image=capture.read()
            if ok:cv2.imwrite(str(out/f'inspect_{index}.png'),image)
        capture.release();print('VIDEO',video,info['result'],info['self_hp'],info['opponent_hp'],flush=True)
    finally:
        if recorder is not None:recorder.release()
        env.close()


if __name__=='__main__':main()
