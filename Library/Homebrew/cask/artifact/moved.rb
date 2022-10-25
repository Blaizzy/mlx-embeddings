# typed: true
# frozen_string_literal: true

require "cask/artifact/relocated"

module Cask
  module Artifact
    # Superclass for all artifacts that are installed by moving them to the target location.
    #
    # @api private
    class Moved < Relocated
      extend T::Sig

      sig { returns(String) }
      def self.english_description
        "#{english_name}s"
      end

      def install_phase(**options)
        move(**options)
      end

      def uninstall_phase(**options)
        move_back(**options)
      end

      def summarize_installed
        if target.exist?
          "#{printable_target} (#{target.abv})"
        else
          Formatter.error(printable_target, label: "Missing #{self.class.english_name}")
        end
      end

      private

      def move(adopt: false, force: false, verbose: false, command: nil, **options)
        unless source.exist?
          raise CaskError, "It seems the #{self.class.english_name} source '#{source}' is not there."
        end

        if Utils.path_occupied?(target)
          if adopt
            ohai "Adopting existing #{self.class.english_name} at '#{target}'"
            same = command.run(
              "/usr/bin/diff",
              args:         ["--recursive", "--brief", source, target],
              verbose:      verbose,
              print_stdout: verbose,
            ).success?

            unless same
              raise CaskError,
                    "It seems the existing #{self.class.english_name} is different from " \
                    "the one being installed."
            end

            # Remove the source as we don't need to move it to the target location
            source.rmtree

            return post_move(command)
          end

          message = "It seems there is already #{self.class.english_article} " \
                    "#{self.class.english_name} at '#{target}'"
          raise CaskError, "#{message}." unless force

          opoo "#{message}; overwriting."
          delete(target, force: force, command: command, **options)
        end

        ohai "Moving #{self.class.english_name} '#{source.basename}' to '#{target}'"
        if target.dirname.ascend.find(&:directory?).writable?
          target.dirname.mkpath
        else
          command.run!("/bin/mkdir", args: ["-p", target.dirname], sudo: true)
        end

        if target.dirname.writable?
          FileUtils.move(source, target)
        else
          command.run!("/bin/mv", args: [source, target], sudo: true)
        end

        post_move(command)
      end

      # Performs any actions necessary after the source has been moved to the target location.
      def post_move(command)
        FileUtils.ln_sf target, source

        add_altname_metadata(target, source.basename, command: command)
      end

      def move_back(skip: false, force: false, command: nil, **options)
        FileUtils.rm source if source.symlink? && source.dirname.join(source.readlink) == target

        if Utils.path_occupied?(source)
          message = "It seems there is already #{self.class.english_article} " \
                    "#{self.class.english_name} at '#{source}'"
          raise CaskError, "#{message}." unless force

          opoo "#{message}; overwriting."
          delete(source, force: force, command: command, **options)
        end

        unless target.exist?
          return if skip || force

          raise CaskError, "It seems the #{self.class.english_name} source '#{target}' is not there."
        end

        ohai "Backing #{self.class.english_name} '#{target.basename}' up to '#{source}'"
        source.dirname.mkpath

        # We need to preserve extended attributes between copies.
        command.run!("/bin/cp", args: ["-pR", target, source], sudo: !target.parent.writable?)

        delete(target, force: force, command: command, **options)
      end

      def delete(target, force: false, command: nil, **_)
        ohai "Removing #{self.class.english_name} '#{target}'"
        raise CaskError, "Cannot remove undeletable #{self.class.english_name}." if MacOS.undeletable?(target)

        return unless Utils.path_occupied?(target)

        if target.parent.writable? && !force
          target.rmtree
        else
          Utils.gain_permissions_remove(target, command: command)
        end
      end
    end
  end
end
