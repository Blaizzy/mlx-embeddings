# frozen_string_literal: true

require "tap"
require "cli/parser"

module Homebrew
  module_function

  def tap_new_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `tap-new` <user>`/`<repo>

        Generate the template files for a new tap.
      EOS

      named 1
    end
  end

  def tap_new
    args = tap_new_args.parse

    tap_name = args.named.first
    tap = Tap.fetch(args.named.first)
    raise "Invalid tap name '#{tap_name}'" unless tap.path.to_s.match?(HOMEBREW_TAP_PATH_REGEX)

    titleized_user = tap.user.dup
    titleized_repo = tap.repo.dup
    titleized_user[0] = titleized_user[0].upcase
    titleized_repo[0] = titleized_repo[0].upcase

    (tap.path/"Formula").mkpath

    readme = <<~MARKDOWN
      # #{titleized_user} #{titleized_repo}

      ## How do I install these formulae?
      `brew install #{tap}/<formula>`

      Or `brew tap #{tap}` and then `brew install <formula>`.

      ## Documentation
      `brew help`, `man brew` or check [Homebrew's documentation](https://docs.brew.sh).
    MARKDOWN
    write_path(tap, "README.md", readme)

    actions = <<~YAML
      name: brew test-bot
      on:
        push:
          branches: master
        pull_request:
      jobs:
        test-bot:
          runs-on: ${{ matrix.os }}
          strategy:
            matrix:
              os: [ubuntu-latest, macOS-latest]
          steps:
            - name: Set up Homebrew
              id: set-up-homebrew
              uses: Homebrew/actions/setup-homebrew@master

            - name: Cache Homebrew Bundler RubyGems
              id: cache
              uses: actions/cache@v1
              with:
                path: ${{ steps.set-up-homebrew.outputs.gems-path }}
                key: ${{ runner.os }}-rubygems-${{ steps.set-up-homebrew.outputs.gems-hash }}
                restore-keys: ${{ runner.os }}-rubygems-

            - name: Install Homebrew Bundler RubyGems
              if: steps.cache.outputs.cache-hit != 'true'
              run: brew install-bundler-gems

            - run: brew test-bot --only-cleanup-before

            - run: brew test-bot --only-setup

            - run: brew test-bot --only-tap-syntax

            - run: brew test-bot --only-formulae
              if: github.event_name == 'pull_request'
    YAML

    (tap.path/".github/workflows").mkpath
    write_path(tap, ".github/workflows/tests.yml", actions)
    ohai "Created #{tap}"
    puts tap.path.to_s
  end

  def write_path(tap, filename, content)
    path = tap.path/filename
    tap.path.mkpath
    raise "#{path} already exists" if path.exist?

    path.write content
  end
end
