# typed: false
# frozen_string_literal: true

require "utils/github/actions"

module Cask
  class Cmd
    # Cask implementation of the `brew audit` command.
    #
    # @api private
    class Audit < AbstractCommand
      extend T::Sig

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

      sig { void }
      def run
        casks = args.named.flat_map do |name|
          next name if File.exist?(name)
          next Tap.fetch(name).cask_files if name.count("/") == 1

          name
        end
        casks = casks.map { |c| CaskLoader.load(c, config: Config.from_args(args)) }
        any_named_args = casks.any?
        casks = Cask.to_a if casks.empty?

        results = self.class.audit_casks(
          *casks,
          download:        args.download?,
          appcast:         args.appcast?,
          online:          args.online?,
          strict:          args.strict?,
          new_cask:        args.new_cask?,
          token_conflicts: args.token_conflicts?,
          quarantine:      args.quarantine?,
          any_named_args:  any_named_args,
          language:        args.language,
        )

        self.class.print_annotations(results)

        failed_casks = results.reject { |_, result| result[:errors].empty? }.map(&:first)
        return if failed_casks.empty?

        raise CaskError, "audit failed for casks: #{failed_casks.join(" ")}"
      end

      def self.audit_casks(
        *casks,
        download: nil,
        appcast: nil,
        online: nil,
        strict: nil,
        new_cask: nil,
        token_conflicts: nil,
        quarantine: nil,
        any_named_args: nil,
        language: nil
      )
        options = {
          audit_download:        download,
          audit_appcast:         appcast,
          audit_online:          online,
          audit_strict:          strict,
          audit_new_cask:        new_cask,
          audit_token_conflicts: token_conflicts,
          quarantine:            quarantine,
          language:              language,
          any_named_args:        any_named_args,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        Homebrew.auditing = true

        require "cask/auditor"

        casks.map do |cask|
          odebug "Auditing Cask #{cask}"
          [cask, Auditor.audit(cask, **options)]
        end.to_h
      end

      def self.print_annotations(results)
        return unless ENV["GITHUB_ACTIONS"]

        results.each do |cask, result|
          cask_path = cask.sourcefile_path
          annotations = (result[:warnings].map { |w| [:warning, w] } + result[:errors].map { |e| [:error, e] })
                        .map { |type, message| GitHub::Actions::Annotation.new(type, message, file: cask_path) }

          annotations.each do |annotation|
            puts annotation if annotation.relevant?
          end
        end
      end
    end
  end
end
