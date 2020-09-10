# frozen_string_literal: true

require "utils/github/actions"

module Cask
  class Cmd
    # Implementation of the `brew cask audit` command.
    #
    # @api private
    class Audit < AbstractCommand
      def self.description
        <<~EOS
          Check <cask> for Homebrew coding style violations. This should be run before
          submitting a new cask. If no <cask> is provided, checks all locally
          available casks. Will exit with a non-zero status if any errors are
          found, which can be useful for implementing pre-commit hooks.
        EOS
      end

      def self.parser
        super do
          switch "--download",
                 description: "Audit the downloaded file"
          switch "--[no-]appcast",
                 description: "Audit the appcast"
          switch "--token-conflicts",
                 description: "Audit for token conflicts"
          switch "--strict",
                 description: "Run additional, stricter style checks"
          switch "--online",
                 description: "Run additional, slower style checks that require a network connection"
          switch "--new-cask",
                 description: "Run various additional style checks to determine if a new cask is eligible " \
                              "for Homebrew. This should be used when creating new casks and implies " \
                              "`--strict` and `--online`"
        end
      end

      def run
        require "cask/auditor"

        Homebrew.auditing = true

        options = {
          audit_download:        args.download?,
          audit_appcast:         args.appcast?,
          audit_online:          args.online?,
          audit_strict:          args.strict?,
          audit_new_cask:        args.new_cask?,
          audit_token_conflicts: args.token_conflicts?,
          quarantine:            args.quarantine?,
          language:              args.language,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        failed_casks = casks(alternative: -> { Cask.to_a })
                       .reject do |cask|
          odebug "Auditing Cask #{cask}"
          result = Auditor.audit(cask, **options)

          next true if result[:warnings].empty? && result[:errors].empty?

          if ENV["GITHUB_ACTIONS"]
            cask_path = cask.sourcefile_path
            annotations = (result[:warnings].map { |w| [:warning, w] } + result[:errors].map { |e| [:error, e] })
                          .map { |type, message| GitHub::Actions::Annotation.new(type, message, file: cask_path) }

            annotations.each do |annotation|
              puts annotation if annotation.relevant?
            end
          end

          false
        end

        return if failed_casks.empty?

        raise CaskError, "audit failed for casks: #{failed_casks.join(" ")}"
      end
    end
  end
end
