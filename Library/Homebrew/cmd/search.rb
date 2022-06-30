# typed: false
# frozen_string_literal: true

require "formula"
require "missing_formula"
require "descriptions"
require "cli/parser"
require "search"

module Homebrew
  extend T::Sig

  module_function

  extend Search

  PACKAGE_MANAGERS = {
    repology:  ->(query) { "https://repology.org/projects/?search=#{query}" },
    macports:  ->(query) { "https://ports.macports.org/search/?q=#{query}" },
    fink:      ->(query) { "https://pdb.finkproject.org/pdb/browse.php?summary=#{query}" },
    opensuse:  ->(query) { "https://software.opensuse.org/search?q=#{query}" },
    fedora:    ->(query) { "https://apps.fedoraproject.org/packages/s/#{query}" },
    archlinux: ->(query) { "https://archlinux.org/packages/?q=#{query}" },
    debian:    lambda { |query|
      "https://packages.debian.org/search?keywords=#{query}&searchon=names&suite=all&section=all"
    },
    ubuntu:    lambda { |query|
      "https://packages.ubuntu.com/search?keywords=#{query}&searchon=names&suite=all&section=all"
    },
  }.freeze

  sig { returns(CLI::Parser) }
  def search_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Perform a substring search of cask tokens and formula names for <text>. If <text>
        is flanked by slashes, it is interpreted as a regular expression.
        The search for <text> is extended online to `homebrew/core` and `homebrew/cask`.
      EOS
      switch "--formula", "--formulae",
             description: "Search online and locally for formulae."
      switch "--cask", "--casks",
             description: "Search online and locally for casks."
      switch "--desc",
             description: "Search for formulae with a description matching <text> and casks with " \
                          "a name or description matching <text>."
      switch "--pull-request",
             description: "Search for GitHub pull requests containing <text>."
      switch "--open",
             depends_on:  "--pull-request",
             description: "Search for only open GitHub pull requests."
      switch "--closed",
             depends_on:  "--pull-request",
             description: "Search for only closed GitHub pull requests."
      package_manager_switches = PACKAGE_MANAGERS.keys.map { |name| "--#{name}" }
      package_manager_switches.each do |s|
        switch s,
               description: "Search for <text> in the given database."
      end

      conflicts "--desc", "--pull-request"
      conflicts "--open", "--closed"
      conflicts(*package_manager_switches)

      named_args :text_or_regex, min: 1
    end
  end

  def search
    args = search_args.parse

    return if search_package_manager(args)

    query = args.named.join(" ")
    string_or_regex = query_regexp(query)

    if args.desc?
      search_descriptions(string_or_regex, args)
    elsif args.pull_request?
      search_pull_requests(query, args)
    else
      search_names(query, string_or_regex, args)
    end

    print_regex_help(args)
  end

  def print_regex_help(args)
    return unless $stdout.tty?

    metacharacters = %w[\\ | ( ) [ ] { } ^ $ * + ?].freeze
    return unless metacharacters.any? do |char|
      args.named.any? do |arg|
        arg.include?(char) && !arg.start_with?("/")
      end
    end

    opoo <<~EOS
      Did you mean to perform a regular expression search?
      Surround your query with /slashes/ to search locally by regex.
    EOS
  end

  def search_package_manager(args)
    package_manager = PACKAGE_MANAGERS.find { |name,| args[:"#{name}?"] }
    return false if package_manager.nil?

    _, url = package_manager
    exec_browser url.call(URI.encode_www_form_component(args.named.join(" ")))
    true
  end

  def search_pull_requests(query, args)
    only = if args.open? && !args.closed?
      "open"
    elsif args.closed? && !args.open?
      "closed"
    end

    GitHub.print_pull_requests_matching(query, only)
  end

  def search_names(query, string_or_regex, args)
    remote_results = search_taps(query, silent: true)

    local_formulae = search_formulae(string_or_regex)
    remote_formulae = remote_results[:formulae]
    all_formulae = local_formulae + remote_formulae

    local_casks = search_casks(string_or_regex)
    remote_casks = remote_results[:casks]
    all_casks = local_casks + remote_casks

    print_formulae = args.formula?
    print_casks = args.cask?
    print_formulae = print_casks = true if !print_formulae && !print_casks
    print_formulae &&= all_formulae.any?
    print_casks &&= all_casks.any?

    ohai "Formulae", Formatter.columns(all_formulae) if print_formulae
    puts if print_formulae && print_casks
    ohai "Casks", Formatter.columns(all_casks) if print_casks

    count = all_formulae.count + all_casks.count

    print_missing_formula_help(query, count.positive?) if local_casks.exclude?(query)

    odie "No formulae or casks found for #{query.inspect}." if count.zero?
  end

  def print_missing_formula_help(query, found_matches)
    return unless $stdout.tty?

    reason = MissingFormula.reason(query, silent: true)
    return if reason.nil?

    if found_matches
      puts
      puts "If you meant #{query.inspect} specifically:"
    end
    puts reason
  end
end
