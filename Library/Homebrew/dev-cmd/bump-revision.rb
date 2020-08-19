# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def bump_revision_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump-revision` [<options>] <formula> [<formula> ...]

        Create a commit to increment the revision of <formula>. If no revision is
        present, "revision 1" will be added.
      EOS
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      flag   "--message=",
             description: "Append <message> to the default commit message."

      min_named :formula
    end
  end

  def bump_revision
    args = bump_revision_args.parse

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    args.named.to_formulae.each do |formula|
      current_revision = formula.revision

      if current_revision.zero?
        formula_spec = formula.stable
        hash_type, old_hash = if (checksum = formula_spec.checksum)
          [checksum.hash_type, checksum.hexdigest]
        end

        old = if formula.license
          license_string = if formula.license.length > 1
            formula.license
          else
            "\"#{formula.license.first}\""
          end
          # insert replacement revision after license
          <<~EOS
            license #{license_string}
          EOS
        elsif formula.path.read.include?("stable do\n")
          # insert replacement revision after homepage
          <<~EOS
            homepage "#{formula.homepage}"
          EOS
        elsif hash_type
          # insert replacement revision after hash
          <<~EOS
            #{hash_type} "#{old_hash}"
          EOS
        else
          # insert replacement revision after :revision
          <<~EOS
            revision: "#{formula_spec.specs[:revision]}"
          EOS
        end
        replacement = "#{old}  revision 1\n"

      else
        old = "revision #{current_revision}"
        replacement = "revision #{current_revision+1}"
      end

      if args.dry_run?
        ohai "replace #{old.inspect} with #{replacement.inspect}" unless args.quiet?
      else
        Utils::Inreplace.inreplace(formula.path) do |s|
          s.gsub!(old, replacement)
        end
      end

      message = "#{formula.name}: revision bump #{args.message}"
      if args.dry_run?
        ohai "git commit --no-edit --verbose --message=#{message} -- #{formula.path}"
      else
        formula.path.parent.cd do
          safe_system "git", "commit", "--no-edit", "--verbose",
                      "--message=#{message}", "--", formula.path
        end
      end
    end
  end
end
