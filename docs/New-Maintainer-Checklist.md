# New Maintainer Checklist

**This is a guide used by existing maintainers to invite new maintainers. You might find it interesting but there's nothing here users should have to know.**

There's someone who has been making consistently high-quality contributions to Homebrew and shown themselves able to make slightly more advanced contributions than just e.g. formula updates? Let's invite them to be a maintainer!

First, send them the invitation email:

```markdown
The Homebrew team and I really appreciate your help on issues, pull requests and
your contributions to Homebrew.

We would like to invite you to have commit access and be a Homebrew maintainer.
If you agree to be a maintainer, you should spend a significant proportion of
the time you are working on Homebrew applying and self-merging widely used
changes (e.g. version updates), triaging, fixing and debugging user-reported
issues, or reviewing user pull requests. You should also be making contributions
to Homebrew at least once per quarter.

You should watch or regularly check Homebrew/brew and/or
Homebrew/homebrew-core. Let us know which (or both) so we can grant you commit
access appropriately.

If you're no longer able to perform all of these tasks, please continue to
contribute to Homebrew, but we will ask you to step down as a maintainer.

A few requests:

- Please make pull requests on any changes to Homebrew/brew code or any
  non-trivial (e.g. not a test or audit improvement or version bump) changes
  to formulae code and don't merge them unless you get at least one approval
  and passing tests.
- Use `brew pull` for formulae changes that require new bottles or change
  multiple formulae and let it auto-close issues wherever possible (it may
  take ~5m). When this isn't necessary use GitHub's "Merge pull request"
  button in "create a merge commit" mode for Homebrew/brew or "squash and
  merge" for a single formulae change. If in doubt, check with e.g. Fork.app
  that you've not accidentally added merge commits.
- Still create your branches on your fork rather than in the main repository.
  Note GitHub's UI will create edits and reverts on the main repository if you
  make edits or click "Revert" on the Homebrew/brew repository rather than your
  own fork.
- If still in doubt please ask for help and we'll help you out.
- Please read:
    - https://docs.brew.sh/Brew-Test-Bot-For-Core-Contributors
    - https://docs.brew.sh/Maintainer-Guidelines
    - anything else you haven't read on https://docs.brew.sh

How does that sound?

Thanks for all your work so far!
```

If they accept, follow a few steps to get them set up:

- Invite them to the [**@Homebrew/maintainers** team](https://github.com/orgs/Homebrew/teams/maintainers) (or any relevant [subteams](https://github.com/orgs/Homebrew/teams/maintainers/teams)) to give them write access to relevant repositories (but don't make them owners). They will need to enable [GitHub's Two Factor Authentication](https://help.github.com/articles/about-two-factor-authentication/).
- Ask them to sign in to [Bintray](https://bintray.com) using their GitHub account and they should auto-sync to [Bintray's Homebrew organisation](https://bintray.com/homebrew/organization/edit/members) as a member so they can publish new bottles.
- Add them to the [Jenkins' GitHub Authorization Settings admin user names](https://jenkins.brew.sh/configureSecurity/) so they can adjust settings and restart jobs.
- Add them to the [Jenkins' GitHub Pull Request Builder admin list](https://jenkins.brew.sh/configure) to enable `@BrewTestBot test this please` for them.
- Invite them to the [`homebrew-maintainers` private maintainers mailing list](https://lists.sfconservancy.org/mailman/admin/homebrew-maintainers/members/add).
- Invite them to the [`machomebrew` private maintainers Slack](https://machomebrew.slack.com/admin/invites) (and ensure they've read the [communication guidelines](Maintainer-Guidelines.md#communication)) and ask them to use their real name there (rather than a pseudonym they may use on e.g. GitHub).
- Invite them to the Homebrew Maintainers Monthly Call on Google calendar.
- Ask them to disable SMS as a 2FA device or fallback on their GitHub account in favour of using one of the other authentication methods.
- Ask them to (regularly) review remove any unneeded [GitHub personal access tokens](https://github.com/settings/tokens).
- Add them to Homebrew/brew's README, run `brew man` and commit the changes.
- Start the process to [add them as Homebrew members](#members), for formal voting rights and the ability to hold office for Homebrew.

If they are interested in doing system administration work or Homebrew/brew releases:

- Invite them to the [`homebrew-ops` private operations mailing list](https://lists.sfconservancy.org/mailman/admin/homebrew-ops/members/add).
- Invite them to the [`homebrew` private 1Password](https://homebrew.1password.com/people).

If they are elected to the Homebrew's [Project Leadership Committee](https://docs.brew.sh/Homebrew-Governance#4-project-leadership-committee):

- Email their name, email and employer to the [Software Freedom Conservancy](https://sfconservancy.org) at homebrew@sfconservancy.org
- Make them [owners on the Homebrew GitHub organisation](https://github.com/orgs/Homebrew/people)
- Invite them to the [**@Homebrew/plc** team](https://github.com/orgs/Homebrew/teams/plc/members)
- Invite them to [Google Analytics](https://analytics.google.com/analytics/web/#management/Settings/a76679469w115400090p120682403/%3Fm.page%3DAccountUsers/) and add them to the relevant section of the [Homebrew/brew's README](https://github.com/Homebrew/brew/edit/master/README.md).
- Invite them to the [`homebrew` private 1Password](https://homebrew.1password.com/people).
- Make them owners on the [`machomebrew` private maintainers Slack](https://machomebrew.slack.com/admin)).

If there are problems, ask them to step down as a maintainer and revoke their access to all of the above.

In interests of loosely verifying maintainer identity and building camaraderie, if you find yourself in the same town (e.g living, visiting or at a conference) as another Homebrew maintainer you should make the effort to meet up. If you do so, you can expense your meal (within [SFC reimbursable expense policies](https://sfconservancy.org/projects/policies/conservancy-travel-policy.html#meals-for-organizational-development). This is a more relaxed version of similar policies used by other projects, e.g. the Debian system to meet in person to sign keys with legal ID verification.

Now sit back, relax and let the new maintainers handle more of our contributions.

## Members

People who are either not eligible or willing to be Homebrew maintainers but have shown continued involvement in the Homebrew community may be admitted by a majority vote of the [Project Leadership Committee](https://docs.brew.sh/Homebrew-Governance#4-project-leadership-committee) to join the Homebrew GitHub organisation as [members](https://docs.brew.sh/Homebrew-Governance#2-members).

When admitted as members:

- Invite them to the [**@Homebrew/members** team](https://github.com/orgs/Homebrew/teams/members), to give them access to the private governance repository.
- Invite them as a single-channel guest to the #members channel on the [`machomebrew` private maintainers Slack](https://machomebrew.slack.com/admin/invites) (and ensure they've read the [communication guidelines](Maintainer-Guidelines.md#communication)) and ask them to use their real name there (rather than a pseudonym they may use on e.g. GitHub).
- Add them to the membership list in the [homebrew-governance repository](https://github.com/Homebrew/homebrew-governance).
