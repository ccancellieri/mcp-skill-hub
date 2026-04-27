#compdef cwt
# zsh completion for cwt (worktree-driven Claude sessions).
# Source from ~/.zshrc:   source /path/to/mcp-skill-hub/scripts/_cwt-completion.zsh

_cwt() {
    local -a projects modes
    # Project candidates: any directory containing a .git under repo_roots.
    local roots
    roots=$(python3 -c "from skill_hub import config; print(' '.join(config.load_config().get('worktree',{}).get('repo_roots',['~/work/code'])))" 2>/dev/null)
    [[ -z "$roots" ]] && roots="~/work/code"
    for root in ${(z)roots}; do
        root=${~root}
        [[ -d "$root" ]] || continue
        for d in "$root"/*(N/); do
            [[ -d "$d/.git" ]] && projects+=("${d:t}")
        done
    done
    modes=(terminal tmux background)

    case "$CURRENT" in
        2)
            if [[ "$words[2]" == --* ]]; then
                _arguments \
                    '--resume[resume task by id]:task id:' \
                    '--list[list open tasks]'
            else
                _describe -t projects 'project' projects
            fi
            ;;
        3) _message 'task name (slug)' ;;
        *)
            _arguments \
                '--mode[session mode]:mode:(terminal tmux background)' \
                '--prompt[initial prompt]:prompt:'
            ;;
    esac
}
compdef _cwt cwt
