# frozen_string_literal: true

require "json"
require "cask/installer"

module Cask
  class Cmd
    class Info < AbstractCommand
      option "--json=VERSION", :json

      def initialize(*)
        super
        raise CaskUnspecifiedError if args.empty?
      end

      def run
        if json == "v1"
          puts JSON.generate(casks.map(&:to_h))
        else
          casks.each_with_index do |cask, i|
            puts unless i.zero?
            odebug "Getting info for Cask #{cask}"
            self.class.info(cask)
          end
        end
      end

      def self.help
        "displays information about the given Cask"
      end

      def self.get_info(cask)
        output = title_info(cask) + "\n"
        output << Formatter.url(cask.homepage) + "\n" if cask.homepage
        output << installation_info(cask)
        repo = repo_info(cask)
        output << repo + "\n" if repo
        output << name_info(cask)
        language = language_info(cask)
        output << language if language
        output << artifact_info(cask) + "\n"
        caveats = Installer.caveats(cask)
        output << caveats if caveats
        output
      end

      def self.info(cask)
        puts get_info(cask)
      end

      def self.title_info(cask)
        title = "#{cask.token}: #{cask.version}"
        title += " (auto_updates)" if cask.auto_updates
        title
      end

      def self.formatted_url(url)
        "#{Tty.underline}#{url}#{Tty.reset}"
      end

      def self.installation_info(cask)
        return "Not installed\n" unless cask.installed?

        install_info = +""
        cask.versions.each do |version|
          versioned_staged_path = cask.caskroom_path.join(version)
          path_details = if versioned_staged_path.exist?
            versioned_staged_path.abv
          else
            Formatter.error("does not exist")
          end
          install_info << "#{versioned_staged_path} (#{path_details})\n"
        end
        install_info.freeze
      end

      def self.name_info(cask)
        <<~EOS
          #{ohai_title((cask.name.size > 1) ? "Names" : "Name")}
          #{cask.name.empty? ? Formatter.error("None") : cask.name.join("\n")}
        EOS
      end

      def self.language_info(cask)
        return if cask.languages.empty?

        <<~EOS
          #{ohai_title("Languages")}
          #{cask.languages.join(", ")}
        EOS
      end

      def self.repo_info(cask)
        return if cask.tap.nil?

        url = if cask.tap.custom_remote? && !cask.tap.remote.nil?
          cask.tap.remote
        else
          "#{cask.tap.default_remote}/blob/master/Casks/#{cask.token}.rb"
        end

        "From: #{Formatter.url(url)}"
      end

      def self.artifact_info(cask)
        artifact_output = ohai_title("Artifacts").dup
        cask.artifacts.each do |artifact|
          next unless artifact.respond_to?(:install_phase)
          next unless DSL::ORDINARY_ARTIFACT_CLASSES.include?(artifact.class)

          artifact_output << "\n" << artifact.to_s
        end
        artifact_output.freeze
      end
    end
  end
end
