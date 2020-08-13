# frozen_string_literal: true

module Cask
  class Cmd
    class List < AbstractCommand
      option "-1",             :one, false
      option "--versions",     :versions, false
      option "--full-name",    :full_name, false
      option "--json",         :json, false

      def self.usage
        <<~EOS
          `cask list`, `cask ls` [<options>] [<casks>]

          -1          - Force output to be one entry per line.
                        This is the default when output is not to a terminal.
          --versions  - Show the version number for installed formulae, or only the specified
                        casks if <casks> are provided.
          --full-name - Print casks with fully-qualified names.
          --json      - Print a JSON representation of <cask>. See the docs for examples of using the JSON
                        output: <https://docs.brew.sh/Querying-Brew>

          List all installed casks.

          If <casks> are provided, limit information to just those casks.
        EOS
      end

      def self.help
        "lists installed Casks or the casks provided in the arguments"
      end

      def run
        self.class.list_casks(
          *casks,
          json:      json?,
          one:       one?,
          full_name: full_name?,
          versions:  versions?,
        )
      end

      def self.list_casks(*casks, json: false, one: false, full_name: false, versions: false)
        output = if casks.any?
          casks.each do |cask|
            raise CaskNotInstalledError, cask unless cask.installed?
          end
        else
          Caskroom.casks
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
          puts output.map(&method(:list_artifacts))
        elsif !output.empty?
          puts Formatter.columns(output.map(&:to_s))
        end
      end

      def self.list_artifacts(cask)
        cask.artifacts.group_by(&:class).each do |klass, artifacts|
          next unless klass.respond_to?(:english_description)

          return "==> #{klass.english_description}", artifacts.map(&:summarize_installed)
        end
      end

      def self.format_versioned(cask)
        cask.to_s.concat(cask.versions.map(&:to_s).join(" ").prepend(" "))
      end
    end
  end
end
