import WorkspaceApp from '@/components/workspace-app'

export default async function RoutePage({ params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params
  return <WorkspaceApp initialSlug={slug[0] ?? 'overview'} />
}
