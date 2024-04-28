# typed: strict
# frozen_string_literal: true

require "abstract_command"
require "formula"
require "tab"

module Homebrew
  module Cmd
    class TabCmd < AbstractCommand
      cmd_args do
        description <<~EOS
          Edit tab information for installed formulae.

          This can be useful when you want to control whether an installed
          formula should be removed by `brew autoremove`.
          To prevent removal, mark the formula as installed on request;
          to allow removal, mark the formula as not installed on request.
        EOS

        switch "--installed-on-request",
               description: "Mark <formula> as installed on request."
        switch "--no-installed-on-request",
               description: "Mark <formula> as not installed on request."

        conflicts "--installed-on-request", "--no-installed-on-request"

        named_args :formula, min: 1
      end

      sig { override.void }
      def run
        installed_on_request = if args.installed_on_request?
          true
        elsif args.no_installed_on_request?
          false
        end
        raise UsageError, "No marking option specified." if installed_on_request.nil?

        formulae = args.named.to_formulae
        if (formulae_not_installed = formulae.reject(&:any_version_installed?)).any?
          formula_names = formulae_not_installed.map(&:name)
          is_or_are = (formula_names.length == 1) ? "is" : "are"
          odie "#{formula_names.to_sentence} #{is_or_are} not installed."
        end

        formulae.each do |formula|
          update_tab formula, installed_on_request:
        end
      end

      private

      sig { params(formula: Formula, installed_on_request: T::Boolean).void }
      def update_tab(formula, installed_on_request:)
        tab = Tab.for_formula(formula)
        unless tab.tabfile.exist?
          raise ArgumentError,
                "Tab file for #{formula.name} does not exist."
        end

        installed_on_request_str = "#{"not " unless installed_on_request}installed on request"
        if (tab.installed_on_request && installed_on_request) ||
           (!tab.installed_on_request && !installed_on_request)
          ohai "#{formula.name} is already marked as #{installed_on_request_str}."
          return
        end

        tab.installed_on_request = installed_on_request
        tab.write
        ohai "#{formula.name} is now marked as #{installed_on_request_str}."
      end
    end
  end
end
