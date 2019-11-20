# frozen_string_literal: true

require "formula"
require "missing_formula"
require "descriptions"
require "cli/parser"
require "search"

module Homebrew
  module_function

  extend Search

  PACKAGE_MANAGERS = {
    macports: ->(query) { "https://www.macports.org/ports.php?by=name&substr=#{query}" },
    fink:     ->(query) { "http://pdb.finkproject.org/pdb/browse.php?summary=#{query}" },
    opensuse: ->(query) { "https://software.opensuse.org/search?q=#{query}" },
    fedora:   ->(query) { "https://apps.fedoraproject.org/packages/s/#{query}" },
    debian:   lambda { |query|
      "https://packages.debian.org/search?keywords=#{query}&searchon=names&suite=all&section=all"
    },
    ubuntu:   lambda { |query|
      "https://packages.ubuntu.com/search?keywords=#{query}&searchon=names&suite=all&section=all"
    },
  }.freeze

  def search_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `search` [<options>] [<text>|`/`<text>`/`]

        Perform a substring search of cask tokens and formula names for <text>. If <text>
        is flanked by slashes, it is interpreted as a regular expression.
        The search for <text> is extended online to `homebrew/core` and `homebrew/cask`.

        If no <text> is provided, list all locally available formulae (including tapped ones).
        No online search is performed.
      EOS
      switch "--casks",
             description: "List all locally available casks (including tapped ones). "\
                          "No online search is performed."
      switch "--desc",
             description: "Search for formulae with a description matching <text> and casks with "\
                          "a name matching <text>."

      package_manager_switches = PACKAGE_MANAGERS.keys.map { |name| "--#{name}" }
      package_manager_switches.each do |s|
        switch s,
               description: "Search for <text> in the given package manager's list."
      end
      switch :verbose
      switch :debug
      conflicts(*package_manager_switches)
    end
  end

  def search
    search_args.parse

    if package_manager = PACKAGE_MANAGERS.find { |name,| args[:"#{name}?"] }
      _, url = package_manager
      exec_browser url.call(URI.encode_www_form_component(args.remaining.join(" ")))
      return
    end

    if args.remaining.empty?
      if args.casks?
        puts Formatter.columns(Cask::Cask.to_a.map(&:full_name).sort)
      else
        puts Formatter.columns(Formula.full_names.sort)
      end

      return
    end

    query = args.remaining.join(" ")
    string_or_regex = query_regexp(query)

    if args.desc?
      search_descriptions(string_or_regex)
    else
      remote_results = search_taps(query, silent: true)

      local_formulae = search_formulae(string_or_regex)
      remote_formulae = remote_results[:formulae]
      all_formulae = local_formulae + remote_formulae

      local_casks = search_casks(string_or_regex)
      remote_casks = remote_results[:casks]
      all_casks = local_casks + remote_casks

      if all_formulae.any?
        ohai "Formulae"
        puts Formatter.columns(all_formulae)
      end

      if all_casks.any?
        puts if all_formulae.any?
        ohai "Casks"
        puts Formatter.columns(all_casks)
      end

      if $stdout.tty?
        count = all_formulae.count + all_casks.count

        if (reason = MissingFormula.reason(query, silent: true)) && !local_casks.include?(query)
          if count.positive?
            puts
            puts "If you meant #{query.inspect} specifically:"
          end
          puts reason
        elsif count.zero?
          puts "No formula or cask found for #{query.inspect}."
          GitHub.print_pull_requests_matching(query)
        end
      end
    end

    return unless $stdout.tty?
    return if args.remaining.empty?

    metacharacters = %w[\\ | ( ) [ ] { } ^ $ * + ?].freeze
    return unless metacharacters.any? do |char|
      args.remaining.any? do |arg|
        arg.include?(char) && !arg.start_with?("/")
      end
    end

    opoo <<~EOS
      Did you mean to perform a regular expression search?
      Surround your query with /slashes/ to search locally by regex.
    EOS
  end
end
