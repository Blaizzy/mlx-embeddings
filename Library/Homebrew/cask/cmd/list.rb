# typed: false
# frozen_string_literal: true

require "cask/artifact/relocated"

module Cask
  class Cmd
    # Cask implementation of the `brew list` command.
    #
    # @api private
    class List < AbstractCommand
      extend T::Sig

      def self.parser
        super do
          switch "-1",
                 description: "Force output to be one entry per line."
          switch "--versions",
                 description: "Show the version number the listed casks."
          switch "--full-name",
                 description: "Print casks with fully-qualified names."
          switch "--json",
                 description: "Print a JSON representation of the listed casks. "
        end
      end

      sig { void }
      def run
        self.class.list_casks(
          *casks,
          json:      args.json?,
          one:       args.public_send(:'1?'),
          full_name: args.full_name?,
          versions:  args.versions?,
          args:      args,
        )
      end

      def self.list_casks(*casks, args:, json: false, one: false, full_name: false, versions: false)
        output = if casks.any?
          casks.each do |cask|
            raise CaskNotInstalledError, cask unless cask.installed?
          end
        else
          Caskroom.casks(config: Config.from_args(args))
        end

        if json
          puts JSON.generate(output.map(&:to_h))
        elsif one
          puts output.map(&:to_s)
        elsif full_name
          puts output.map(&:full_name).sort(&tap_and_name_comparison)
        elsif versions
          puts output.map(&method(:format_versioned))
        elsif !output.empty? && casks.any?
          output.map(&method(:list_artifacts))
        elsif !output.empty?
          puts Formatter.columns(output.map(&:to_s))
        end
      end

      def self.list_artifacts(cask)
        cask.artifacts.group_by(&:class).sort_by { |klass, _| klass.english_name }.each do |klass, artifacts|
          next if [Artifact::Uninstall, Artifact::Zap].include? klass

          ohai klass.english_name
          artifacts.each do |artifact|
            puts artifact.summarize_installed if artifact.respond_to?(:summarize_installed)
            next if artifact.respond_to?(:summarize_installed)

            puts artifact
          end
        end
      end

      def self.format_versioned(cask)
        cask.to_s.concat(cask.versions.map(&:to_s).join(" ").prepend(" "))
      end
    end
  end
end
