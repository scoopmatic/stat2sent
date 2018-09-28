import json
import numpy as np
import random
import re


def stats2intro_sampler(sp, vocab, batch_size=10, input_len=60, output_len=15):
    """ Data generator producing pairs of statistics text input and report intro output """

    #story_article_map = json.load(open("story_article_map.json"))
    #story_stats_map = json.load(open("story_gamestats_map.json"))
    #stat_report_map = json.load(open("report_sent_ids.json"))
    game_report_intros = json.load(open("gamereport_intros.json"))
    game_stat_texts = json.load(open("gamestat_texts.json"))

    j = 0
    while True:
        batch_ok = True
        batch_input = np.zeros((batch_size, input_len))
        batch_output = np.zeros((batch_size, output_len))
        batch_output_shifted = np.zeros((batch_size, output_len))
        #for b, story in enumerate(random.sample(story_article_map.keys(), batch_size)):
        #for b, stat in enumerate(random.sample(stat_report_map.keys(), batch_size)):
        for b, stat in enumerate(random.sample(game_report_intros.keys(), batch_size)):
            """try:
                stat = random.sample(story_stats_map[story], 1)[0]
            except KeyError:
                continue
            art = random.sample(story_article_map[story], 1)[0]
            art = art[art.index('/')+1:].replace('.xml', '.conll')"""
            try:
                report = random.sample(game_report_intros[stat], 1)
                report = report[0]
            except ValueError:
                continue

            i = 0
            input = game_stat_texts[stat]
            #print("Input:",input[:input_len])
            #input = sp.EncodeAsIds(input)[:input_len]

            #print(input.split('\n')[0])
            teams = input[input.index('[TEAM:')+6:]
            teams = teams[:teams.index(']')]
            if '–' in teams:
                teams = teams.lower().split('–')
            elif '-' in teams:
                teams = teams.lower().split('-')
            else:
                continue
            #print(teams)
            input = re.sub(r"\[SCORE:(\d+)–(\d+)\]", r"[SCORE: \1 – \2 ]", input)
            #input = input.split()
            #input = [x for x in input if x.isdigit()][:2]
            #input = [vocab[token] for token in input[:input_len]]
            #input = sp.EncodeAsIds(' '.join(input))[:input_len]
            input = sp.EncodeAsIds(input)[:input_len]
            #import pdb; pdb.set_trace()
            batch_input[b,:len(input)] = input
            #print(' '.join([sp.IdToPiece(i) for i in input]))
            #output_ids = [sp[token] for token in report[:output_len]]
            report = ["[TEAM1:%s]" % teams[0] if w.lower().startswith("[team:%s" % teams[0]) else w for w in report]
            try:
                report = ["[TEAM2:%s]" % teams[1] if w.lower().startswith("[team:%s" % teams[1]) else w for w in report]
            except IndexError:
                import pdb; pdb.set_trace()

            #print(report)
            #print()
            output = ' '.join(report)
            #output = re.sub(r"\[SCORE:([A-Za-z0-9:–\-]+)\.?\]", r"\1", output)
            #output = re.sub(r"\[PERSON:([A-Za-z\-#]+)\]", r"\1", output)
            output = re.sub(r"\[TEAM:([A-Za-z\-#]+)\]", r"[TEAM?:\1]", output)
            #print(output)
            output_ids = sp.EncodeAsIds(output)
            #print(' '.join([sp.IdToPiece(i) for i in output_ids]))
            output_ids = output_ids[:output_len]

            batch_output[b,:len(output_ids)] = output_ids
            batch_output_shifted[b,1:len(output_ids)] = output_ids[:-1]
            batch_output_shifted[b,0] = 1
            # art -> sent_seq -> sent_seq_docvecs

        if batch_ok:
            yield ([batch_input, batch_output_shifted], batch_output.reshape((batch_output.shape[0], batch_output.shape[1], 1)))


def stats2docvecs_sampler(sp, d2v, batch_size=10, input_len=60, output_len=15):
    """ Data generator producing pairs of statistics text input and sentence vector sequence output """

    #story_article_map = json.load(open("story_article_map.json"))
    #story_stats_map = json.load(open("story_gamestats_map.json"))
    stat_report_map = json.load(open("report_sent_ids.json"))
    game_stat_texts = json.load(open("gamestat_texts.json"))

    docvec_size = d2v.docvecs[0].shape[0]
    j = 0
    while True:
        batch_ok = True
        batch_input = np.zeros((batch_size, input_len))
        batch_output = np.zeros((batch_size, output_len, docvec_size))
        batch_output_shifted = np.zeros((batch_size, output_len))
        #for b, story in enumerate(random.sample(story_article_map.keys(), batch_size)):
        for b, stat in enumerate(random.sample(stat_report_map.keys(), batch_size)):
            """try:
                stat = random.sample(story_stats_map[story], 1)[0]
            except KeyError:
                continue
            art = random.sample(story_article_map[story], 1)[0]
            art = art[art.index('/')+1:].replace('.xml', '.conll')"""
            try:
                report = random.sample(stat_report_map[stat], 1)
                report = report[0]
            except ValueError:
                continue

            i = 0
            vecs = []
            #while ('%s_%d' % (art, i)) in d2v.docvecs:
            #while ('%s' % (j)) in d2v.docvecs:
                #vecs.append(d2v.docvecs['%s_%d' % (art, i)])
                #i += 1
            #if ('%s' % (j)) in d2v.docvecs:
            for sent_id in report:
                vecs.append(d2v.docvecs['%s' % (sent_id)])
            else:
                #print(j, "not found in docvecs")
                pass
            j = (j + 1) % len(d2v.docvecs)
            #print(stat, art)
            """input = open("../latka_txt/%s.txt" % stat).read()
            if not input:
                batch_ok = False
                print(stat, "bad")
                continue
            """
            input = game_stat_texts[stat]
            #print("Input:",input[:input_len])
            #input = sp.EncodeAsIds(input)[:input_len]
            input = re.sub(r"\[SCORE:(\d+)–(\d+)\]", r"[SCORE: \1 – \2 ]", input)
            input = input.split()
            input = [sp[token] for token in input[:input_len]]
            #import pdb; pdb.set_trace()
            batch_input[b,:len(input)] = input
            #print(' '.join([sp.IdToPiece(i) for i in input]))
            output = vecs[:output_len]
            if not output:
                batch_ok = False
                break
            batch_output[b,:len(output)] = output
            output_ids = report[:output_len]
            batch_output_shifted[b,1:len(output_ids)] = output_ids[:-1]
            batch_output_shifted[b,1:len(output_ids)] += 1
            # art -> sent_seq -> sent_seq_docvecs

        if batch_ok:
            yield ([batch_input, batch_output_shifted], batch_output)
