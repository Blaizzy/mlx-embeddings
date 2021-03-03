# typed: true
# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def dispatch_build_bottle_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Build bottles for these formulae with GitHub Actions.
      EOS
      flag   "--tap=",
             description: "Target tap repository (default: `homebrew/core`)."
      flag   "--issue=",
             description: "If specified, post a comment to this issue number if the job fails."
      flag   "--macos=",
             description: "Version of macOS the bottle should be built for."
      flag   "--workflow=",
             description: "Dispatch specified workflow (default: `dispatch-build-bottle.yml`)."
      switch "--upload",
             description: "Upload built bottles to Bintray."
      switch "--linux",
             description: "Dispatch bottle for Linux (using GitHub runners)."

      conflicts "--macos", "--linux"
      named_args :formula, min: 1
    end
  end

  def dispatch_build_bottle
    args = dispatch_build_bottle_args.parse

    tap = Tap.fetch(args.tap || CoreTap.instance.name)
    user, repo = tap.full_name.split("/")
    ref = "master"
    workflow = args.workflow || "dispatch-build-bottle.yml"

    # Ensure we dispatch the bottle in homebrew/homebrew-core
    # TODO: remove when core taps are merged
    repo.gsub!("linux", "home") unless args.tap

    if (macos = args.macos)
      # Fixup version for ARM/Apple Silicon
      # TODO: fix label name to be 11-arm64 instead and remove this.
      macos.gsub!(/^11-arm$/, "11-arm64")

      macos = macos.yield_self do |s|
        MacOS::Version.from_symbol(s.to_sym)
      rescue MacOSVersionError
        MacOS::Version.new(s)
      end

      # Fixup label for ARM/Apple Silicon
      macos_label = if macos.arch == :arm64
        # TODO: fix label name to be 11-arm64 instead.
        "#{macos}-arm"
      else
        macos.to_s
      end

      dispatching_for = "macOS #{macos}"
    elsif T.unsafe(args).linux?
      workflow = args.workflow || "linux-#{workflow}"
      dispatching_for = "Linux"
    else
      raise UsageError, "Must specify --macos or --linux option"
    end

    args.named.to_resolved_formulae.each do |formula|
      # Required inputs
      inputs = {
        formula: formula.name,
      }

      # Optional inputs
      # These cannot be passed as nil to GitHub API
      inputs[:macos] = macos_label if args.macos
      inputs[:issue] = args.issue if args.issue
      inputs[:upload] = args.upload?.to_s if args.upload?

      ohai "Dispatching #{tap} bottling request of formula \"#{formula.name}\" for #{dispatching_for}"
      GitHub.workflow_dispatch_event(user, repo, workflow, ref, inputs)
    end
  end
end
